import numpy as np

def manual_rgb2gray(rgb_img):
    """Manual RGB to Grayscale conversion using the Luminosity Method."""
    if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 3:
        r, g, b = rgb_img[..., 0], rgb_img[..., 1], rgb_img[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        if gray.max() > 1.0:
            gray = gray / 255.0
        return gray
    return rgb_img

def manual_convolve2d(image, kernel):
    """Manual 2D Convolution (Cross-correlation) using NumPy array slicing."""
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(image, dtype=np.float64)
    
    for i in range(k_h):
        for j in range(k_w):
            out += padded[i:i+image.shape[0], j:j+image.shape[1]] * kernel[i, j]
    return out

def manual_gaussian_filter(image, sigma):
    """Manual Gaussian Blur utilizing Mathematical Separability."""
    if sigma <= 1e-6:
        return image.copy()
        
    size = int(6 * sigma)
    size = size + 1 if size % 2 == 0 else size
    size = max(3, size)
    
    k = size // 2
    x = np.arange(-k, k + 1)
    
    g1d = np.exp(-(x**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()
    
    kernel_x = g1d.reshape(1, -1)
    kernel_y = g1d.reshape(-1, 1)
    
    img_x = manual_convolve2d(image, kernel_x)
    return manual_convolve2d(img_x, kernel_y)

def normalize(arr):
    """Normalize array strictly to [0, 1] bounds."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)