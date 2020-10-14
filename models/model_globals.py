FLAG_TF = (True, False, 1, 0)

class GLOBAL(object):
    TORCH_ALIGN_CORNERS = False
    # TORCH_BATCH_NORMAL_TRACK = False
    # TORCH_INSTANCE_NORMAL_TRACK = False
    TORCH_RELU_INPLACE = False

    @staticmethod
    def torch_align_corners(flag=None):
        if ( flag is None ):
            return GLOBAL.TORCH_ALIGN_CORNERS
        else:
            assert( flag in FLAG_TF )
            GLOBAL.TORCH_ALIGN_CORNERS = flag

    # @staticmethod
    # def torch_batch_normal_track_stat(flag=None):
    #     if ( flag is None ):
    #         return GLOBAL.TORCH_BATCH_NORMAL_TRACK
    #     else:
    #         assert( flag in FLAG_TF )
    #         GLOBAL.TORCH_BATCH_NORMAL_TRACK = flag

    # @staticmethod
    # def torch_inst_normal_track_stat(flag=None):
    #     if ( flag is None ):
    #         return GLOBAL.TORCH_INSTANCE_NORMAL_TRACK
    #     else:
    #         assert( falg in FLAG_TF )
    #         GLOBAL.TORCH_INSTANCE_NORMAL_TRACK = flag

    @staticmethod
    def torch_relu_inplace(flag=None):
        if ( flag is None ):
            return GLOBAL.TORCH_RELU_INPLACE
        else:
            assert( flag in FLAG_TF )
            GLOBAL.TORCH_RELU_INPLACE = flag