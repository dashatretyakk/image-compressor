
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_image(path, size=(512, 512)):
    # Load image in grayscale
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Resize to the specified size
    image = cv2.resize(image, size)
    return image

image_path = '3.jpg'
image = load_image(image_path)

def apply_dct(image, block_size=8):
    h, w = image.shape
    dct_image = np.zeros_like(image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            # Apply DCT to the block and store it in dct_image
            dct_block = cv2.dct(np.float32(block))
            dct_image[i:i+block_size, j:j+block_size] = dct_block
    return dct_image

dct_image = apply_dct(image)

def apply_haar_wavelet(image):
    # Apply Haar wavelet transform using downsampling
    haar_wavelet = cv2.pyrDown(image)  # Downsampling to simulate Haar transform
    return haar_wavelet

haar_image = apply_haar_wavelet(image)

def inverse_dct(dct_image, block_size=8):
    h, w = dct_image.shape
    reconstructed_image = np.zeros_like(dct_image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_image[i:i+block_size, j:j+block_size]

            idct_block = cv2.idct(block)
            reconstructed_image[i:i+block_size, j:j+block_size] = idct_block
    return reconstructed_image

reconstructed_dct_image = inverse_dct(dct_image)
reconstructed_haar_image = cv2.pyrUp(haar_image)

psnr_dct = psnr(image.astype(np.float32), reconstructed_dct_image, data_range=255)
psnr_haar = psnr(image.astype(np.float32), reconstructed_haar_image, data_range=255)

print("PSNR for DCT compression:", psnr_dct)
print("PSNR for Haar Wavelet compression:", psnr_haar)
