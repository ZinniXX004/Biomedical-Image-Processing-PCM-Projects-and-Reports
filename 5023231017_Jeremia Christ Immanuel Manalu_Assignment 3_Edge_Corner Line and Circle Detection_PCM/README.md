# 🔬 Advanced Medical Image Detector: Edge, Corner, Line & Circle

## 📖 Project Overview
This project is an advanced, modular GUI application built with **PyQt6** and **Python** for the Medical Image Processing (*Pengolahan Citra Medika*) course. It visualizes classical and modern feature extraction algorithms.

The core feature of this application is the **Manual Mathematical Implementation**—bypassing high-level libraries to implement operations natively using pure `NumPy` array slicing and matrix mathematics.

## ✨ Key Features & Assignments

### 📚 Assignment 1: Fundamental Processing
* **Pre-processing:** Luminosity RGB-to-Grayscale conversion, Gaussian Denoising (Separable 1D Convolutions).
* **Enhancement:** Contrast Stretching, Histogram Equalization (HE), CLAHE.
* **Traditional Gradients:** Custom Convolution for Prewitt, Sobel, Roberts, Extended Sobel, and Kirsch compass operators.
* **Evaluation:** Live RMSE, PSNR, SSIM, and Shannon Entropy metrics.

### 📚 Assignment 2: Canny Edge & Sharpening
* **Canny Edge Detection (From Scratch):** Includes Gaussian Gradients, Angle Quantization, Non-Maximum Suppression (NMS), and Hysteresis Double Thresholding.
* **Image Sharpening:** Laplacian Operators ($H_4, H_8, H_{12}$) and Unsharp Masking (USM).
* **Analysis:** Real-time **Intensity Profile** to observe overshoot/undershoot artifacts on edges.

### 📚 Assignment 3: Feature Extraction (Corner, Line, Circle)
* **Harris Corner Detection:** Structure tensor matrix ($M$) generation, Gaussian smoothing, and $Q$-Value scoring. Includes Q-Map Distribution evaluation.
* **Hough Line Transform:** Hessian Normal Form mapping ($r = x\cos(\theta) + y\sin(\theta)$) into a Log-scale Accumulator Heatmap.
* **Hough Circle Transform:** Parametric circle mapping with dynamic radius thresholding.
* **Comprehensive Benchmarking:** Runtime (ms) bar charts and side-by-side visual grids comparing "From Scratch" vs "Library (`skimage`)" implementations.

## 📂 Modular Architecture
To ensure scalability and clean code, the application uses an MVC-inspired modular architecture:

```text
medical_imaging_app/
│
├── main.py                     # Application entry point
├── config.py                   # UI Colors, Kernel Matrices, Constants
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── core/                       # 🧠 BACKEND / CORE LOGIC
│   ├── __init__.py
│   ├── math_ops.py             # Pure Math: Convolve, Gaussian, RGB2Gray
│   ├── enhancement.py          # HE, CLAHE, Contrast Stretching, Metrics
│   ├── edge_detection.py       # Canny (Scratch & Lib), Kirsch, Sobel
│   ├── sharpening.py           # Laplacian, Unsharp Masking
│   ├── feature_extraction.py   # Harris, Hough Line, Hough Circle
│   └── pipeline.py             # Pipeline orchestrator & benchmarking
│
└── gui/                        # 🖥️ FRONTEND INTERFACE
    ├── __init__.py
    ├── components.py           # Custom PyQt6 Widgets (Image/Graph Panels)
    └── main_window.py          # Main layout, event handlers, and tab routing
```

## 🚀 How to Run
**1. Install Requirements:**

Make sure you have Python 3.8++ installed, then run:
```bash
pip install -r requirements.txt
```
**2. Launch the Application:**

Execute the main entry point:
```bash
python main.py
```
**3. Usage Flow:**
*   Load any medical/facial image using the "📂 Load Facial Image" button.

*   Navigate through the tabs (Assignment 1, 2, and 3) at the top.

*   Adjust sliders on the Left Panel (e.g., Gaussian σ, Thresholds, Harris α, Hough θ steps) to see real-time mathematically driven updates across all panels.

## 👨‍💻 Developer Notes

By separating the core mathematical processing from the gui components, the application is highly robust. The UI functions simply as a consumer of the dictionaries produced by the backend pipelines, ensuring a non-blocking and responsive user experience even during heavy Accumulator Space (Hough Transform) computations.