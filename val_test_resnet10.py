# evaluate_and_test.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load the trained model
model = load_model("trained_model.h5")

# Data generator configuration for testing (similar to training configuration)
test_batch_size = 16  # Adjust the batch size for testing if needed

test_datagen = ImageDataGenerator(rescale=1./255)

# Path to your test dataset
test_data_dir = 'path/to/test_dataset'

# Load the test dataset using the data generator
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(84, 84),  # Adjust according to the desired dimensions
    batch_size=test_batch_size,
    class_mode='categorical',
    shuffle=False  # Set to False to maintain the order of images for evaluation
)

# Evaluate the model on the test dataset
eval_result = model.evaluate(test_generator)

# Print the evaluation result (accuracy and loss)
print("Test Accuracy:", eval_result[1])
print("Test Loss:", eval_result[0])

# Alternatively, you can make predictions on individual images
# Load an image for testing
img_path = 'path/to/single_test_image.jpg'
img = image.load_img(img_path, target_size=(84, 84))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions
predictions = model.predict(img_array)

# Print the predicted class probabilities
print("Predicted Probabilities:", predictions)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)
print("Predicted Class Index:", predicted_class_index)
