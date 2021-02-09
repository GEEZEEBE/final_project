import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import layers
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

image_size = (224, 224)
train_path = 'dataset/train'
validation_path = 'dataset/validation'
trained_model_path = "new_trained_from_mobilenet_jun"

n_class = 2
batch_size = 64
epochs = 50

train_datagen = ImageDataGenerator(
        # rotation_range=180,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        # vertical_flip=True,
        # validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        # class_mode='binary',
        shuffle=True,
        # subset='training'
)

validation_generator = train_datagen.flow_from_directory(
        validation_path,
        target_size=image_size,
        batch_size=batch_size,
        # class_mode='binary',
        shuffle=True,
        # subset='validation'
)


conv_layers = mobilenet_v2.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(image_size[0], image_size[1], 3)
)

model = tf.keras.models.Sequential()
model.add(conv_layers)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(n_class, activation='softmax'))
print(model.summary())

model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
)

history = model.fit(
        train_generator,
        batch_size=train_generator.batch_size,
        validation_data=validation_generator,
        epochs=epochs,
        verbose=1
)

model.save(trained_model_path)
