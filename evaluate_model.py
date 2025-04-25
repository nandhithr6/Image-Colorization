import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    # Load colored image (ground truth)
    color_img = load_img(image_path, target_size=target_size)
    color_array = img_to_array(color_img)
    
    # Create grayscale version
    gray_img = load_img(image_path, color_mode='grayscale', target_size=target_size)
    gray_array = img_to_array(gray_img)
    gray_array = np.repeat(gray_array, 3, axis=-1)
    
    # Save grayscale version for visualization
    plt.imsave('grayscale_input.jpg', np.squeeze(gray_array/255.0), cmap='gray')
    
    # Normalize images
    color_array = color_array / 255.0
    gray_array = gray_array / 255.0
    
    return gray_array, color_array

def evaluate_colorization(model, test_image_path):
    # Load and preprocess the test image
    input_array, ground_truth = load_and_preprocess_image(test_image_path)
    
    # Add batch dimension
    input_batch = np.expand_dims(input_array, axis=0)
    
    # Generate colorized output
    predicted = model.predict(input_batch)
    predicted = np.squeeze(predicted, axis=0)
    
    # Calculate metrics
    ssim_value = ssim(ground_truth, predicted, channel_axis=2, data_range=1.0)
    psnr_value = psnr(ground_truth, predicted, data_range=1.0)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(np.squeeze(input_array[:,:,0]), cmap='gray')
    plt.title('Input (Grayscale)')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(predicted)
    plt.title('Predicted (Colorized)')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(ground_truth)
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.suptitle(f'SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f} dB')
    plt.savefig('evaluation_results.png')
    plt.close()
    
    # Save the colorized output
    predicted_uint8 = (predicted * 255).astype(np.uint8)
    colorized_image = array_to_img(predicted_uint8)
    colorized_image.save('colorized_output.jpg')
    
    return ssim_value, psnr_value

# Load the model
model = tf.keras.models.load_model('colorization_model.keras')

# Test image path
test_image = '3.jpg'  # Using the new colored image

# Evaluate the model
ssim_score, psnr_score = evaluate_colorization(model, test_image)

print(f"\nEvaluation Results:")
print(f"SSIM Score: {ssim_score:.4f}")
print(f"PSNR Score: {psnr_score:.4f} dB")
print("\nVisualization has been saved as 'evaluation_results.png'")
print("Colorized output has been saved as 'colorized_output.jpg'")

# Interpretation of results
print("\nInterpretation:")
print("SSIM (Structural Similarity Index):")
print("- Ranges from -1 to 1 (1 being perfect structural similarity)")
print("- Values above 0.7 generally indicate good quality")
print("\nPSNR (Peak Signal-to-Noise Ratio):")
print("- Higher values indicate better quality")
print("- Values above 30dB generally indicate good quality")
print("- Values above 40dB indicate excellent quality") 