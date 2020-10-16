
from models import GLOBAL
# GLOBAL.torch_align_corners(True)
GLOBAL.torch_align_corners(False)

import argparse
import cv2
from models import hsm
import numpy as np
import os
import pdb
import skimage.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from models.submodule import *
from utils.eval import mkdir_p, save_pfm
from utils.preprocess import get_transform
#cudnn.benchmark = True
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='HSM')
parser.add_argument('datapath', 
                    help='test data path')
parser.add_argument('--file-list', type=str, default='Files.csv', 
                    help='The file list. ')
parser.add_argument('--pc-q', type=str, default='', 
                    help='The Q matrix for point cloud reconstruction. ')
parser.add_argument('--loadmodel', default=None,
                    help='model path')
parser.add_argument('--outdir', default='output',
                    help='output dir')
parser.add_argument('--clean', type=float, default=-1,
                    help='clean up output using entropy estimation')
parser.add_argument('--testres', type=float, default=0.5,
                    help='test time resolution ratio 0-x')
parser.add_argument('--max-disp', type=float, default=1024,
                    help='maximum disparity to search for')
parser.add_argument('--level', type=int, default=1,
                    help='output level of output, default is level 1 (stage 3),\
                          can also use level 2 (stage 2) or level 3 (stage 1)')
parser.add_argument('--max-num', type=int, default=0, 
                    help='The maximum number of files to process. Debug use. Set 0 to disable.')
args = parser.parse_args()

assert(args.max_disp > 0)

# # dataloader
# from dataloader import listfiles as DA
# test_left_img, test_right_img, _, _ = DA.dataloader(args.datapath)

# Customized system packages.
from StereoDataTools import file_access, file_list
test_left_img, test_right_img, dispFnList, maskFnList = \
    file_list.read_file_lists(args.file_list, args.datapath)

# print(test_left_img)

from StereoDataTools import metric
from StereoDataTools import point_cloud

# construct model
model = hsm(128,args.clean,level=args.level)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# dry run
multip = 48
imgL = np.zeros((1,3,24*multip,32*multip))
imgR = np.zeros((1,3,24*multip,32*multip))
imgL = Variable(torch.FloatTensor(imgL).cuda())
imgR = Variable(torch.FloatTensor(imgR).cuda())
with torch.no_grad():
    model.eval()
    pred_disp,entropy = model(imgL,imgR)

def main():
    processed = get_transform()
    
    model.eval()

    ## change max disp
    max_disp = int(args.max_disp)
    tmpdisp = int(max_disp*args.testres//64*64)
    if (max_disp*args.testres/64*64) > tmpdisp:
        model.module.maxdisp = tmpdisp + 64
    else:
        model.module.maxdisp = tmpdisp
    if model.module.maxdisp ==64: model.module.maxdisp=128
    model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
    model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()
    print(model.module.maxdisp)

    # Point cloud.
    if ( '' != args.pc_q ):
        # Load the Q matrix.
        Q = np.loadtxt(args.pc_q, dtype=np.float32)
    else:
        Q = None

    nFiles = len(test_left_img)
    nFiles = args.max_num if nFiles > args.max_num > 0 else nFiles

    metrics = np.zeros((nFiles, 9), dtype=np.float32)
    for inx in range(nFiles):
        print('%d/%d: %s. ' % (inx+1, nFiles, test_left_img[inx]))
        imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))[:,:,:3]
        imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))[:,:,:3]
        imgsize = imgL_o.shape[:2]

        # resize
        imgL_o = cv2.resize(imgL_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(imgR_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()

        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h-imgL.shape[2]
        left_pad = max_w-imgL.shape[3]
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        # test
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_disp,entropy = model(imgL,imgR)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time)
            # print('time = %.2f' % (ttime*1000) )
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad   = max_h-imgL_o.shape[0]
        left_pad  = max_w-imgL_o.shape[1]
        entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]

        # resize to highres
        pred_disp = cv2.resize(pred_disp/args.testres,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
        pred_disp[invalid] = np.inf

        # ========== Save. ==========

        # save predictions
        outDir = '%s/%03d' % (args.outdir, inx)
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        np.save(os.path.join(outDir, 'disp.npy'), pred_disp)
        np.save(os.path.join(outDir, 'ent.npy'), entropy)
        cv2.imwrite(os.path.join(outDir, 'disp.png'), pred_disp/pred_disp[~invalid].max()*255)
        cv2.imwrite(os.path.join(outDir, 'ent.png'),entropy/entropy.max()*255)

        with open(os.path.join(outDir, 'time.txt'),'w') as f:
             f.write(str(ttime))

        # Metrics.
        maskValid = np.isfinite( pred_disp )
        pred_disp = np.expand_dims( pred_disp, axis=0 )
        dispT = file_access.read_float( dispFnList[inx] ) if ( dispFnList[inx] != 'None' ) else None
        if ( dispT is not None ):
            dispT = np.expand_dims( dispT, axis=0 )

            maskT = file_access.read_mask( maskFnList[inx] ) if ( maskFnList[inx] != 'None' ) else None
            if ( maskT is not None ):
                maskT = np.logical_and( maskValid, maskT )
            else:
                maskT = maskValid

            maskT = np.expand_dims( maskT, axis=0 )

            metrics[inx, :] = metric.epe( dispT, pred_disp, maskT )

        # Point Cloud.
        if ( Q is not None ): 
            # Prepare the RGB source.
            imgOri = cv2.imread(test_left_img[inx], cv2.IMREAD_UNCHANGED)
            imgPC = cv2.cvtColor( imgOri, cv2.COLOR_BGR2RGB )

            # Write PLY file.
            outFn = os.path.join( outDir, 'Full.ply' )
            point_cloud.write_PLY( outFn, np.squeeze(pred_disp, axis=0), Q, 
                flagFlip=False, distLimit=10.0, 
                mask=maskValid, color=imgPC )

            # print('Write PLY file to %s. ' % (outFn))

        torch.cuda.empty_cache()
    
    # Report the metrics.
    outFn = os.path.join(args.outdir, 'Report_EPE.csv')
    metric.report_epe( outFn, metrics )

    # Save the file list.
    outFn = os.path.join(args.outdir, 'EvalFileListImg0.txt')
    np.savetxt(outFn, test_left_img, fmt='%s')

if __name__ == '__main__':
    import TorchCUDAMem
    with TorchCUDAMem.TorchTracemalloc() as tt:
        main()

