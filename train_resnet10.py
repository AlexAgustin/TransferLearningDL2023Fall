import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to build the ResNet10 model
def build_resnet10(input_shape, num_classes):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Blocks 2-4
    for _ in range(3):
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    # Global average pooling
    model.add(GlobalAveragePooling2D())

    # Fully connected layer
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Model configuration
input_shape = (84, 84, 3)  # Adjust according to the input image dimensions
num_classes = 64  # Adjust according to the number of classes in your dataset

# Call the function to build the model
model = build_resnet10(input_shape, num_classes)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data generator configuration
data_dir = 'C:/Users/alexm/Documents/Cuarto/DL/EuroSAT_RGB/selected'  # Adjust the path to your image directory
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(84, 84),  # Adjust according to the desired dimensions
    batch_size=batch_size,
    class_mode='categorical'
)

# Model training
model.fit(train_generator, epochs=10, steps_per_epoch=len(train_generator))

# Save the trained model
model.save("trained_model.h5")
