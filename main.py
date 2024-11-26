import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import pywt
import matplotlib.pyplot as plt

def load_image(path, size=(512, 512)):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    return image


def apply_dct(image, block_size=8, quantization_factor=10):
    h, w = image.shape
    dct_image = np.zeros_like(image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_block = cv2.dct(np.float32(block))
            # Квантування
            dct_block = np.round(dct_block / quantization_factor) * quantization_factor
            dct_image[i:i+block_size, j:j+block_size] = dct_block
    return dct_image

def inverse_dct(dct_image, block_size=8):
    h, w = dct_image.shape
    reconstructed_image = np.zeros_like(dct_image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_image[i:i+block_size, j:j+block_size]
            idct_block = cv2.idct(block)
            reconstructed_image[i:i+block_size, j:j+block_size] = idct_block
    return reconstructed_image


def apply_haar_wavelet(image, level=1):
    coeffs = pywt.wavedec2(image, 'haar', level=level)
    # Квантування кожного коефіцієнта
    coeffs_quantized = [coeffs[0]] + [
        tuple(np.round(c / 10) * 10 for c in coeff) for coeff in coeffs[1:]
    ]
    return coeffs_quantized


def inverse_haar_wavelet(coeffs):
    reconstructed_image = pywt.waverec2(coeffs, 'haar')
    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)


image_path = '3.jpg'
image = load_image(image_path)


dct_image = apply_dct(image)
reconstructed_dct_image = inverse_dct(dct_image)


cv2.imwrite('reconstructed_dct.jpg', reconstructed_dct_image)

# Haar Wavelet
haar_coeffs = apply_haar_wavelet(image, level=2)
reconstructed_haar_image = inverse_haar_wavelet(haar_coeffs)


cv2.imwrite('reconstructed_haar.jpg', reconstructed_haar_image)

# PSNR обчислення
psnr_dct = psnr(image, reconstructed_dct_image, data_range=255)
psnr_haar = psnr(image, reconstructed_haar_image, data_range=255)

print("PSNR for DCT compression:", psnr_dct)
print("PSNR for Haar Wavelet compression:", psnr_haar)


plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("DCT Reconstructed")
plt.imshow(reconstructed_dct_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Haar Reconstructed")
plt.imshow(reconstructed_haar_image, cmap='gray')

plt.show()

