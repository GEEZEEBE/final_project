from tensorflow.python.compiler.tensorrt import trt_convert as trt

input_saved_model_dir = "new_trained_from_resnet50"
output_saved_model_dir = "trt_resnet50"

print('Converting to TF-TRT FP32...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16,
                                                               max_workspace_size_bytes=8000000000)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir,
                                    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir=output_saved_model_dir)
print('Done Converting to TF-TRT FP32')