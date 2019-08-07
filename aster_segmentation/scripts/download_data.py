import os
import shutil
try:
    from tensorflow.python.keras.utils.data_utils import get_file
except ImportError:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras._impl.keras.utils.data_utils import get_file


"https://aster-segmentation.s3-us-west-2.amazonaws.com/15_frame_batch/Aster_fullsize_2D.npz"


if not os.path.exists("../datasets"):
    os.mkdir("../datasets")

path = "Aster_fullsize_2D.npz"
path = get_file(path, origin="https://aster-segmentation.s3-us-west-2.amazonaws.com/15_frame_batch/Aster_fullsize_2D.npz")

#print(path)
shutil.move(path, "../datasets/Aster_fullsize_2D.npz")


if not os.path.exists("../saved_networks"):
    os.mkdir("../saved_networks")

path = "conv_model_std_61.h5"
path = get_file(path, origin="https://aster-segmentation.s3-us-west-2.amazonaws.com/15_frame_batch/trained_networks/conv_model_std_61.h5")

shutil.move(path, "../saved_networks/conv_model_std_61.h5")



path = "conv_model_std_61.npz"
path = get_file(path, origin="https://aster-segmentation.s3-us-west-2.amazonaws.com/15_frame_batch/trained_networks/conv_model_std_61.npz")

shutil.move(path,"../saved_networks/conv_model_std_61.npz")



