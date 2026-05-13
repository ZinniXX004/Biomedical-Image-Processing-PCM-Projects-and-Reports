# Medical-Grade rPPG Heart Rate Monitor

This project is a modern, modular remote Photoplethysmography (rPPG) system built with Python and PyQt6. It extracts blood volume pulse signals directly from a live webcam feed using Computer Vision and Digital Signal Processing (DSP) to estimate Heart Rate (BPM) in real-time.

## 📂 Project Directory Structure

```text
rppg_project/
│
├── main.py                     # Entry point and PyQt6 GUI Application
├── config.py                   # Global default configuration parameters
├── chrom_justification.py      # Standalone scientific evaluation script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── core/                       # Core algorithms (Vision, DSP, Math)
│   ├── __init__.py
│   ├── webcam.py               # Camera I/O handling
│   ├── detector.py             # Face and Eye detection (Haar Cascade)
│   ├── roi_extractor.py        # Spatial averaging and buffering
│   ├── signal_proc.py          # Temporal filtering (BPF and Savitzky-Golay)
│   ├── rppg.py                 # rPPG extraction (CHROM / Green)
│   └── hr_estimator.py         # FFT, Peak Detection, and Kalman Filter Tracking
│
└── monitor/                    # Background rendering and Data logging
    ├── __init__.py
    ├── plotter.py              # Real-time multi-ROI matplotlib rendering
    └── data_saver.py           # Comprehensive CSV and PNG exporter
```

---

## 🧩 Detailed Module Breakdown and Concepts

### 1. Root Directory Scripts

#### `main.py`
The entry point of the application. It features a modern, medical-grade dark-themed GUI built with PyQt6.
*   **Architecture:** Uses a strict Separation of Concerns (SoC) by offloading heavy computations to background threads (`MonitorWorker` and `JustifyWorker`).
*   **UI/UX:** Responsive layouts, dynamic CSS styling, hover effects, and real-time BPM cards with trend indicators.

#### `config.py`
Contains the `DEFAULT_CONFIG` dictionary. This acts as the baseline configuration for camera settings, buffer lengths, filter cutoff frequencies, and Kalman filter parameters.

#### `chrom_justification.py`
A standalone scientific tool designed to justify the use of the CHROM method over standard Single-Channel (Red/Green/Blue) or Independent Component Analysis (ICA) approaches.
*   **Concept:** It computes the Fast Fourier Transform (FFT) and calculates the Signal-to-Noise Ratio (SNR) around the human pulse band (0.75 - 2.5 Hz). It generates a comprehensive 5-panel infographic to prove that CHROM effectively mitigates specular reflection (ambient light artifacts).

---

### 2. `core/` (Core Algorithms)

#### `core/webcam.py`
Manages the hardware camera interface using OpenCV (`cv2.VideoCapture`). It enforces resolution and frames per second (FPS) stability, which is crucial for accurate temporal frequency analysis.

#### `core/detector.py`
Utilizes OpenCV's Viola-Jones Haar Cascade Classifiers (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`) to detect the user's face and eyes. Histogram equalization is applied beforehand to improve detection in challenging lighting.

#### `core/roi_extractor.py`
*   **Concept:** Defines three Regions of Interest (ROI): Forehead, Left Cheek, and Right Cheek based on proportional bounding box coordinates.
*   **Execution:** Extracts spatial averages of the R, G, and B channels per frame, pushing them into a rolling `deque` buffer to form a continuous time-series signal. Supports 3 operational modes: Forehead Only, Cheeks Only, and Combined.

#### `core/signal_proc.py`
Handles 1D temporal signal purification.
*   **IIR Biquad Bandpass Filter:** Focuses the signal on the human heart rate frequency band (0.75 Hz to 2.5 Hz, corresponding to 45–150 BPM). It implements zero-phase filtering (forward-backward filtering) to prevent phase shifting.
*   **Savitzky-Golay Smoothing:** A digital filter that smoothes the signal without distorting signal tendency.
    *   *Math:* It performs a local polynomial regression (of degree $k$) on a series of values (of at least $k+1$ points) to determine the smoothed value for each point using a pseudo-inverse Vandermonde matrix ($J^T J$ inversion).

#### `core/rppg.py`
Transforms the filtered RGB time-series into a single 1D blood volume pulse (BVP) waveform.
*   **CHROM Method (Chrominance-Based):**
    *   *Math:* Projects the RGB channels into two orthogonal chrominance signals assuming a standardized skin-color profile.
    *   $X_s = 3R_n - 2G_n$
    *   $Y_s = 1.5R_n + G_n - 1.5B_n$
    *   The final rPPG signal is computed as $S = X_s - \alpha Y_s$, where $\alpha = \frac{\sigma(X_s)}{\sigma(Y_s)}$. This alpha ratio specifically neutralizes the specular reflection (white light variations) caused by subject movement.

#### `core/hr_estimator.py`
Extracts the final Beats Per Minute (BPM) using multiple approaches.
*   **FFT Analysis:** Uses a Hanning window and zero-padding (next power of 2) before computing the Discrete Fourier Transform. Extracts the frequency with the highest magnitude within the valid heart rate band.
*   **Peak Detection:** Time-domain validation using `scipy.signal.find_peaks` to calculate Inter-Beat Intervals (IBI).
*   **Kalman Filter Tracking:** A 2-State Constant-Velocity Kalman Filter designed to stabilize the BPM readings and prevent erratic jumps.
    *   *States:* $x = [BPM, \text{Trend}]^T$
    *   *Adaptive Noise:* The measurement noise covariance ($R$) dynamically adjusts based on the FFT's confidence score ($R_{adaptive} = R / \text{confidence}$). A low confidence signal forces the Kalman filter to trust its internal prediction rather than the noisy measurement.

---

### 3. `monitor/` (Visualization and Logging)

#### `monitor/plotter.py`
A highly optimized visualization engine using Matplotlib's `FigureCanvasAgg`.
*   **Architecture:** It renders a 12-subplot dashboard (2400x1500px) entirely inside a background thread. The resulting RGB array is then passed to the main thread and displayed using OpenCV. This completely bypasses the traditional GUI locking issues associated with `plt.show()` or `plt.pause()`.

#### `monitor/data_saver.py`
The logging engine triggered by the "Export" button. It generates a comprehensive dataset of the current session:
1.  `rppg_data.csv`: Raw, BPF, BPF+SG, and rPPG values for all channels.
2.  `plot_rppg_all.png`: Time-domain rPPG with peak annotations.
3.  `plot_fft_overlay.png`: Frequency-domain spectrum overlay of all modes.
4.  `plot_bpf_rgb.png`: Before/After filtering comparisons.
5.  `plot_hr_comparison.png`: Bar charts and quantitative metric tables.
6.  `plot_sg_effect.png`: Step-by-step visualization of the Savitzky-Golay effect.
7.  `plot_channel_fft.png`: Isolated FFT spectrums of R, G, and B channels.
8.  `plot_kalman_history.png`: A timeline tracking the Raw FFT vs Median vs Kalman estimates throughout the session.

---

## 🚀 How to Run

1.  **Install Dependencies:**
    Ensure that have Python 3.10+ installed. Run the following command in the terminal:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Application:**
    Run the main script to launch the GUI.
    ```bash
    python main.py
    ```

3.  **Perform CHROM Justification:**
    After letting the camera run for at least 10 seconds to fill the signal buffer, click the **"📊 Justify CHROM"** button in the UI. The system will process the data in the background and automatically open the resulting scientific infographic using the OS's default image viewer.