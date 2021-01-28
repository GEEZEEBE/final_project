import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import sys
from time import time, gmtime, strftime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

trained_model_path = str(sys.argv[1])

t0 = time()

# model = tf.saved_model.load(trained_model_path)
model = tf.keras.models.load_model(trained_model_path)

t = gmtime(time() - t0)
print("Model Load Done in", strftime("%H:%M:%S", t))
