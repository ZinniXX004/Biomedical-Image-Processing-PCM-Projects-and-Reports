# Interactive Edge Detection Viewer

A highly interactive, modular, multi-tabbed desktop application for Edge Detection and Image Enhancement. Built with **Python, PyQt6, Matplotlib, and Scikit-Image**, this tool allows users to visualize every step of the computer vision pipeline in real-time.

## 🌟 Key Features

* **Multi-Tab Interface:** Logically separates the pipeline into Acquisition, Enhancement, Restoration, Gradient Detection, and Results.
* **Real-time Parametric Control:** Adjust Gaussian Sigma ($\sigma$), Edge Thresholding, and Enhancement parameters (CLAHE clip limits, Contrast Stretching percentiles) via interactive sliders.
* **Performance Evaluation:** Dynamically compares enhancement methods using objective metrics (RMSE, PSNR, SSIM, and Shannon Entropy).
* **Statistical Analysis:** Automatically generates Overlaid Histograms, Ogives (Cumulative Distribution Functions), and Row Intensity Profiles.
* **Secret Showcase Mode:** A hidden tab (`✧`) that hides the entire UI to showcase custom desktop wallpaper/background unobtrusively.

---

## 🧮 Mathematical Background and Methodologies

This application visually demonstrates several core digital image processing mathematical concepts.

### 1. Image Enhancement
* **Contrast Stretching (CS):** 
  Linearly scales the pixel intensities between a defined lower ($P_{low}$) and upper ($P_{high}$) percentile to span the full $[0, 1]$ range.
  $$I_{out} = \max\left(0, \min\left(1, \frac{I_{in} - P_{low}}{P_{high} - P_{low}}\right)\right)$$
* **Histogram Equalization (HE):** Flattens the histogram using the Cumulative Distribution Function (CDF) to uniformize intensity distribution.
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Prevents the over-amplification of noise found in standard HE by clipping the histogram at a predefined `clip_limit` before calculating the CDF in localized grid patches.

### 2. Restoration (Gaussian Denoising)
Applies a 2D Gaussian blur to remove high-frequency noise before edge detection. Controlled by the standard deviation ($\sigma$) slider:
$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

### 3. Gradient and Edge Detection
The image ($I$) is convolved ($*$) with specific mathematical kernels ($K_x$ and $K_y$) (e.g., Sobel, Prewitt, Roberts) to approximate spatial derivatives.

* **Horizontal and Vertical Gradients:**
  $$G_x = I * K_x \quad \text{and} \quad G_y = I * K_y$$
* **Edge Magnitude (Strength):**
  $$M = \sqrt{G_x^2 + G_y^2}$$
* **Edge Direction (Angle):**
  $$\theta = \arctan2(G_y, G_x)$$

*(Note: The **Kirsch** operator uses 8 compass-direction kernels instead of 2, taking the maximum response across all 8 directions).*

### 4. Evaluation Metrics
* **RMSE (Root Mean Square Error):** Measures pixel-wise differences. Lower is better.
* **PSNR (Peak Signal-to-Noise Ratio):** Measures image fidelity. Higher is better (measured in dB).
* **SSIM (Structural Similarity Index):** Measures perceived structural change. Max value is $1.0$.
* **Shannon Entropy:** Measures the average information content/randomness in the image.

---

## 📂 Modular File Structure

The monolithic code has been refactored into a clean, maintainable architecture:

* `main.py` — The entry point that initializes the PyQt6 application and loads the main window.
* `config.py` — Contains global constants, the Steam-inspired color palette, and the convolution kernel matrices (Sobel, Prewitt, etc.).
* `processing.py` — Pure mathematical and image processing logic utilizing `scikit-image` and `scipy`.
* `components.py` — Custom PyQt6 widgets, including `ImagePanel`, `AnalysisPanel`, and the high-definition `BackgroundWidget`.
* `main_window.py` — The primary GUI layout, event listeners, and tab management.

---

## ⚙️ Installation and Setup

1. **Ensure Python is installed:** Need Python 3.9 or higher.
2. **Clone/Download the directory:** Ensure all modular `.py` files and the background image are in the same folder.
3. **Install Dependencies:** Open the terminal in the project directory and run:

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

To start the application, simply execute the `main.py` file from the terminal:

```bash
python main.py
```

### Tips for Customization
To change the background image, open `main_window.py`, locate the `_build_ui` method, and replace the filename in `self.custom_bg_path` with the name of the desired image file. Ensure the image is placed in the same root directory.