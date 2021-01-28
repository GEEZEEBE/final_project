import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = resnet50.ResNet50()
model.save('test_resnet50_saved_model')


print('Converting to TF-TRT FP32...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
                                                               max_workspace_size_bytes=8000000000)

converter = trt.TrtGraphConverterV2(input_saved_model_dir='test_resnet50_saved_model',
                                    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='test_resnet50_saved_model_TFTRT_FP32')
print('Done Converting to TF-TRT FP32')


filename = './dog2.jpg'
img = load_img(filename, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
x = tf.constant(x)

input_saved_model = 'test_resnet50_saved_model_TFTRT_FP32'
saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
print(signature_keys)

infer = saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)

labeling = infer(x)
preds = labeling['predictions'].numpy()
print('{} - Predicted: {}'.format(filename, decode_predictions(preds, top=3)[0]))
