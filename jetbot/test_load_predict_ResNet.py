from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing.image import load_img
from IPython.display import display
from tensorflow.keras.preprocessing.image import img_to_array

model = resnet50.ResNet50()
filename = './dog2.jpg'
image = load_img(filename)
display(image)
image = load_img(filename, target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, 224, 224, 3))
image = resnet50.preprocess_input(image)
yhat = model.predict(image)
label = resnet50.decode_predictions(yhat)
label = label[0][0]
print('%s (%.2f%%)' % (label[1], label[2]*100))