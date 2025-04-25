import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# Load the model
model = tf.keras.models.load_model('colorization_model.keras')

# Function to preprocess the input image
def preprocess_image(image_path):
    # Resize the image to the expected input size
    img = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    # Convert the image to an array and duplicate the grayscale channel to create three channels
    img_array = img_to_array(img)
    img_array = np.repeat(img_array, 3, axis=-1)  # Repeat the grayscale channel to create 3 channels
    # Normalize the image
    img_array = img_array / 255.0
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to postprocess the output image
def postprocess_image(output_array):
    # Remove the batch dimension
    output_array = np.squeeze(output_array, axis=0)
    # Denormalize the image
    output_array = (output_array * 255).astype(np.uint8)
    # Convert array to image
    output_img = array_to_img(output_array)
    return output_img

# Test the model with a grayscale image
input_image_path = '2.jpg'  # Updated with your image path
output_image_path = 'colorized_output.jpg'

# Preprocess the input image
input_image = preprocess_image(input_image_path)

# Predict the colorized image
colorized_image_array = model.predict(input_image)

# Postprocess the output image
colorized_image = postprocess_image(colorized_image_array)

# Save the colorized image
colorized_image.save(output_image_path)

print(f'Colorized image saved to {output_image_path}') 