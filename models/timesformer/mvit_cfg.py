from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.IMAGE_SIZE = 224
_C.DATA.NUM_FRAMES = 16
_C.DATA.INPUT_CHANNEL_NUM = [3]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.NUM_CLASSES = 400
_C.MODEL.DROPOUT_RATE = 0.5
_C.MODEL.HEAD_ACT = "none"  # "softmax"


# -----------------------------------------------------------------------------
# MViT options
# -----------------------------------------------------------------------------
_C.MVIT = CfgNode()
_C.MVIT.MODE = "conv"   # Options include `conv`, `max`.
_C.MVIT.POOL_FIRST = False
_C.MVIT.CLS_EMBED_ON = True
_C.MVIT.PATCH_KERNEL = [3, 7, 7]
_C.MVIT.PATCH_STRIDE = [2, 4, 4]
_C.MVIT.PATCH_PADDING = [1, 3, 3]
_C.MVIT.PATCH_2D = False
_C.MVIT.EMBED_DIM = 96
_C.MVIT.NUM_HEADS = 1
_C.MVIT.MLP_RATIO = 4.0
_C.MVIT.QKV_BIAS = True
_C.MVIT.DROPPATH_RATE = 0.2
_C.MVIT.DEPTH = 16
_C.MVIT.NORM = "layernorm"

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = None

# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = [1, 8, 8]

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = [[0, 1, 1, 1], [1, 1, 2, 2], [2, 1, 1, 1], [3, 1, 2, 2], [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [
    7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 2, 2], [15, 1, 1, 1]]

# If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
# Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
_C.MVIT.POOL_KVQ_KERNEL = [3, 3, 3]

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = False

# If True, use norm after stem.
_C.MVIT.NORM_STEM = False

# If True, perform separate positional embedding.
_C.MVIT.SEP_POS_EMBED = False

# Dropout rate for the MViT backbone.
_C.MVIT.DROPOUT_RATE = 0.0

# If True, use absolute positional embedding.
_C.MVIT.USE_ABS_POS = False

# If True, use relative positional embedding for spatial dimentions
_C.MVIT.REL_POS_SPATIAL = True

# If True, use relative positional embedding for temporal dimentions
_C.MVIT.REL_POS_TEMPORAL = True

# If True, init rel with zero
_C.MVIT.REL_POS_ZERO_INIT = False

# If True, using Residual Pooling connection
_C.MVIT.RESIDUAL_POOLING = True

# Dim mul in qkv linear layers of attention block instead of MLP
_C.MVIT.DIM_MUL_IN_ATT = True

# If True, using separate linear layers for Q, K, V in attention blocks.
_C.MVIT.SEPARATE_QKV = False


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
