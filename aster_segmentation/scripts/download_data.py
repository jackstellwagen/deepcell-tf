try:
    from tensorflow.python.keras.utils.data_utils import get_file
except ImportError:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras._impl.keras.utils.data_utils import get_file


"https://aster-segmentation.s3-us-west-2.amazonaws.com/15_frame_batch/Aster_fullsize_2D.npz"

path = "../datasets/Aster_fullsize_2D.npz"
path = get_file(path, origin="https://aster-segmentation.s3-us-west-2.amazonaws.com/15_frame_batch/Aster_fullsize_2D.npz")
