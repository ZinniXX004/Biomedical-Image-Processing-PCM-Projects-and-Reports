# 🔬 Interactive Medical Image Edge Detection Viewer

## 📖 Project Overview
This project is an advanced, modular GUI application built with **PyQt6** and **Python** for the Medical Image Processing (Pengolahan Citra Medika) course. It visualizes classical and modern edge detection, image enhancement, and sharpening algorithms. 

The core feature of this application is the **Manual Mathematical Implementation**—bypassing high-level libraries to implement operations like Convolution, Gaussian Blurring, and Canny Edge Detection natively using pure `NumPy` mathematics.

## ✨ Key Features (Perfect for PPT Slides)
*   **From-Scratch Mathematical Backend:**
    *   Manual 2D Cross-correlation / Convolution via NumPy array slicing.
    *   Mathematical Separability for 1D/2D Gaussian Filters (optimized performance).
    *   Luminosity Method for RGB-to-Grayscale conversion.
*   **Comprehensive Canny Edge Detection (Step-by-Step):**
    *   Gaussian smoothing → Gradients ($f_x, f_y$) → Magnitude & Angle → Angle Quantization (4 directions) → Non-Maximum Suppression (NMS) → Double Thresholding → Hysteresis Edge Tracking.
*   **Image Enhancements & Metrics:**
    *   Contrast Stretching, Histogram Equalization (HE), and CLAHE.
    *   Live evaluation metrics: **RMSE, PSNR, SSIM, and Shannon Entropy**.
*   **Image Sharpening Techniques:**
    *   Laplacian Operators ($H_4, H_8, H_{12}$) and Unsharp Masking (USM).
*   **Modern UI & Analytics:**
    *   Steam-inspired Dark UI theme with real-time Ogive (CDF) & Histogram charts.
    *   Benchmarking Tab to compare the runtime (ms) of 9 different algorithms simultaneously.

## 📂 Project Architecture (Modular Design)
```text
root/
│
├── main.py                     # Application entry point
├── config.py                   # UI Colors, Kernel Matrices, Configurations
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── core/                       # 🧠 CORE PROCESSING LOGIC
│   ├── __init__.py
│   ├── math_ops.py             # Pure Math: Convolve, Gaussian, RGB2Gray
│   ├── enhancement.py          # HE, CLAHE, Contrast Stretching, Metrics
│   ├── edge_detection.py       # Canny (Manual & Skimage), Kirsch, Sobel
│   ├── sharpening.py           # Laplacian, Unsharp Masking
│   └── pipeline.py             # Pipeline orchestrator & benchmarking
│
└── gui/                        # 🖥️ FRONTEND INTERFACE
    ├── __init__.py
    ├── components.py           # Custom PyQt6 Widgets (Image/Graph Panels)
    └── main_window.py          # Main layout, event handlers, and routing
```

## 🚀 How to Run
**1. Install Requirements:**

Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
**2. Launch the Application:**
```bash
python main.py
```
**3. Usage Flow:**
*   Tab 1 - 5 (Assignment 1): Focuses on basic acquisition, enhancement, restoration, traditional gradients (Prewitt, Sobel, Kirsch), and thresholding.

*   Tab 6 - 8 (Assignment 2): Focuses on in-depth Canny Edge Detection, Image Sharpening, and Performance Benchmarking.

*   Left Panel: Adjust configuration parameters (Sigma, Threshold, Weights) in real-time.

## 👨‍💻 Developer Notes

By separating the core mathematical logic from the gui components, the application is highly scalable. The UI simply acts as a consumer to the data dictionary produced by the backend pipelines, ensuring a non-blocking and highly responsive user experience.