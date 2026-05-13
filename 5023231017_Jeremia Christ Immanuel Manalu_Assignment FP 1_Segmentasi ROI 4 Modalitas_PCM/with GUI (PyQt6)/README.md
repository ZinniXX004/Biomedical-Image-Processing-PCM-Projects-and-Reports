# ⟁ Advanced Medical Image Processing Suite

```console
[SYS_INIT] > Booting diagnostic sequence...
[STATUS]   > Multi-modal imaging pipelines online.
```

> **Extracting the invisible from the visible.**
> A PyQt6-driven analytical engine engineered to segment, restore, and extract radiomic features across four distinct biomedical imaging modalities.

---

## ⚙️ SYSTEM ARCHITECTURE

This repository houses a modular, high-performance image processing suite designed for clinical parameter extraction. It utilizes a Model-View-Controller (MVC) architecture, decoupling complex tensor operations from the `PyQt6` interface, ensuring zero-latency rendering and preventing buffer overflow during intensive `matplotlib` plotting.

### Core Modules / Protocols:

#### `[PROTOCOL 01] X-Ray Pneumonia`
* **Target:** Chest Radiograph (CXR).
* **Operation:** Heart segmentation and Cardiothoracic Ratio (CTR) measurement.
* **Technique:** Centroid-based thoracic estimation, CLAHE contrast optimization, and GLCM (Gray-Level Co-occurrence Matrix) texture analysis to penetrate severe pneumonia infiltrates (silhouette sign).

#### `[PROTOCOL 02] CT-Scan Thorax (TIFF & DICOM)`
* **Target:** Axial Computed Tomography slices.
* **Operation:** Cardiac silhouette isolation and physical radiodensity extraction.
* **Technique:** Processes both 8-bit TIFF and raw DICOM matrices. Implements precise Mediastinal Windowing (clamping Hounsfield Units) and extracts absolute physical dimensions (Cardiothoracic Area Ratio - CTAR) using DICOM pixel spacing metadata.

#### `[PROTOCOL 03] Fundus Retinography`
* **Target:** Retinal Fundus Photography.
* **Operation:** Optic Disc (OD) localization and shape restoration.
* **Technique:** Spectral decomposition (Red/Green channel isolation), morphological scalpel (opening/hole-filling) to strip intersecting vasculature, and elliptical bounding parameter extraction.

#### `[PROTOCOL 04] 2D Ultrasound`
* **Target:** Fetal Sonography.
* **Operation:** Cranial biometry extraction (BPD, OFD, HC).
* **Technique:** Non-linear speckle noise suppression via Median filtering, acoustic echo categorization using Multi-Otsu, and advanced mathematical ellipse fitting (utilizing Ramanujan's approximation for Head Circumference).

---

## 💻 INSTALLATION & INITIALIZATION

Ensure your environment is running Python 3.8+ before initializing the sequence.

```bash
# 1. Clone the repository
$git clone [https://github.com/BioLensBME/Medical-Imaging-Suite.git$](https://github.com/BioLensBME/Medical-Imaging-Suite.git$) cd Medical-Imaging-Suite

# 2. Deploy dependencies
$ pip install -r requirements.txt

# 3. Execute the mainframe GUI
$ python main_GUI.py
```

---

## 🖥️ INTERFACE & USAGE

The `main_GUI.py` acts as the central nervous system.
1. Navigate via the top modal tabs (`Fundus Retina`, `Ultrasound`, `Chest X-Ray`, `CT-Scan`).
2. Load the dataset directory using the control panel on the left.
3. Follow the strict hierarchical sequence: **Evaluate Enhancement** -> **Segment & Extract**.
4. Extracted radiomic profiles and physical measurements will be dynamically rendered in the internal Pandas DataModel viewer.

---

## 👤 AUTHOR

**Jeremia Christ Immanuel Manalu**
*Biomedical Engineering | Institut Teknologi Sepuluh Nopember (ITS)*