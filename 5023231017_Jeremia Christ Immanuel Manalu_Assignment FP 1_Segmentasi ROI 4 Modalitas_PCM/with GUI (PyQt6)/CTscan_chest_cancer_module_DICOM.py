'''
Final Project 1 
- Using CT image (DICOM file format) for heart segmentation from Thorax/Chest section
- GUI-Ready Module: Refactored to return Matplotlib Figures and Pandas DataFrames instead of blocking with plt.show() or input()
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom # REQUIRED FOR DICOM FILES
import scipy.ndimage as ndi
from scipy.stats import skew, kurtosis
from skimage import measure, morphology, filters, exposure
from skimage.feature import graycomatrix, graycoprops
import math

# Use white style for a clean background
sns.set_theme(style="white")

def view_all_dicoms(dicom_dir):
    """
    Reads all DICOM files, sorts them by anatomical sequence (InstanceNumber),
    and opens an Interactive Interactive Slice Viewer.
    User can scroll through slices using Mouse Scroll Wheel or Arrow Keys.
    """
    print(f"\n[INFO] Scanning directory: {dicom_dir}")
    files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm') or f.endswith('.dicom') or '.' not in f]
    
    if not files:
        print("[ERROR] No DICOM files found in the specified directory.")
        return []

    print(f"[INFO] Found {len(files)} potential DICOM files. Loading and sorting by anatomical sequence...")
    
    slices = []
    for f in files:
        path = os.path.join(dicom_dir, f)
        try:
            dcm = pydicom.dcmread(path)
            # Default to 0 if sequence metadata is missing
            instance_num = int(dcm.get('InstanceNumber', 0)) 
            slices.append((instance_num, dcm, f))
        except Exception:
            continue

    if not slices:
        print("[ERROR] Could not read any valid DICOM pixel data.")
        return []

    # Sort slices based on anatomical scan sequence (Instance Number)
    slices.sort(key=lambda x: x[0])
    
    images = [s[1].pixel_array for s in slices]
    valid_dicoms = [s[2] for s in slices]
    
    # INTERACTIVE SCROLLABLE MATPLOTLIB VIEWER
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("INTERACTIVE DICOM VIEWER - CLOSE TO PROCEED")
    
    ax.volume = images
    ax.index = len(images) // 2 
    ax.filenames = valid_dicoms

    img_display = ax.imshow(ax.volume[ax.index], cmap='bone')
    ax.set_title(f"Index [{ax.index}] | File: {ax.filenames[ax.index]}\nScroll Mouse UP/DOWN or Use ARROW KEYS", fontsize=12, fontweight='bold')
    ax.axis('off')

    def update_slice():
        # Update image and title when index changes
        img_display.set_array(ax.volume[ax.index])
        ax.set_title(f"Index [{ax.index}] | File: {ax.filenames[ax.index]}\nScroll Mouse UP/DOWN or Use ARROW KEYS", fontsize=12, fontweight='bold')
        fig.canvas.draw()

    def process_scroll(event):
        # Handle mouse wheel movement
        if event.button == 'up':
            ax.index = (ax.index + 1) % len(ax.volume)
        elif event.button == 'down':
            ax.index = (ax.index - 1) % len(ax.volume)
        update_slice()

    def process_key(event):
        # Handle keyboard arrow key movement
        if event.key in ['right', 'up']:
            ax.index = (ax.index + 1) % len(ax.volume)
        elif event.key in ['left', 'down']:
            ax.index = (ax.index - 1) % len(ax.volume)
        update_slice()

    fig.canvas.mpl_connect('scroll_event', process_scroll)
    fig.canvas.mpl_connect('key_press_event', process_key)

    print("[INFO] Displaying Interactive DICOM Viewer.")
    print("[INFO] Scroll to find the best 2 slices, note their Index numbers.")
    print("[INFO] CLOSE THE PLOT WINDOW TO CONTINUE...")
    
    plt.tight_layout()
    plt.show() # Kept here as it acts as an independent popup dialog for selection
    
    return valid_dicoms


class CTProcessor:
    def __init__(self, dicom_dir, selected_files, export_dir):
        """Initialize CT dataset with selected DICOM files and load metadata."""
        self.dataset = {}
        self.metadata = {}
        
        for f in selected_files:
            path = os.path.join(dicom_dir, f)
            dcm = pydicom.dcmread(path)
            
            # 1. EXTRACT DICOM METADATA
            intercept = dcm.get('RescaleIntercept', 0)
            slope = dcm.get('RescaleSlope', 1)
            
            # Extract Pixel Spacing (mm per pixel) for real-world area calculation
            pixel_spacing = dcm.get('PixelSpacing', [1.0, 1.0])
            pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
            
            # 2. CONVERT TO HOUNSFIELD UNITS (HU)
            hu_image = dcm.pixel_array * slope + intercept
            
            self.dataset[f] = hu_image
            self.metadata[f] = {
                "intercept": intercept,
                "slope": slope,
                "pixel_area_cm2": pixel_area_mm2 / 100.0, # Convert mm^2 to cm^2
                "pixel_spacing_x_mm": pixel_spacing[0]
            }
            
        self.export_dir = export_dir
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def apply_ct_windowing(self, hu_image, window_level=40, window_width=400):
        """
        Apply Mediastinal/Soft Tissue Window to isolate the heart.
        WL=40, WW=400 is the gold standard for chest soft tissue.
        """
        img_min = window_level - (window_width / 2.0)
        img_max = window_level + (window_width / 2.0)
        
        windowed_img = np.clip(hu_image, img_min, img_max)
        # Normalize to 0-255 for standard image processing algorithms (Multi-Otsu)
        windowed_img = ((windowed_img - img_min) / window_width) * 255.0
        return windowed_img.astype(np.uint8)

    def extract_features(self, heart_mask, hu_image, thorax_mask, meta):
        # Extract Morphological and Radiomics features using Real-World Units (HU, cm^2, mm)
        labeled_heart = measure.label(heart_mask)
        props = measure.regionprops(labeled_heart, intensity_image=hu_image)
        
        if not props:
            return {}
        
        prop = props[0]
        
        # 1. MORPHOLOGY (REAL-WORLD UNITS)
        # Cardiac Area in cm^2
        cardiac_area_px = prop.area
        cardiac_area_cm2 = cardiac_area_px * meta["pixel_area_cm2"]
        
        # Thoracic Area in cm^2 (Total cavity area inside ribs)
        thorax_area_px = np.sum(thorax_mask)
        thorax_area_cm2 = thorax_area_px * meta["pixel_area_cm2"]
        
        # CTAR (Cardiothoracic Area Ratio) - Superior to 1D CTR
        ctar = cardiac_area_cm2 / thorax_area_cm2 if thorax_area_cm2 > 0 else 0
        
        # Axial CTR (1D Width Ratio)
        minr_h, minc_h, maxr_h, maxc_h = prop.bbox
        cardiac_width_mm = (maxc_h - minc_h) * meta["pixel_spacing_x_mm"]
        
        # Thoracic width from thorax mask
        labeled_thorax = measure.label(thorax_mask)
        thorax_props = measure.regionprops(labeled_thorax)
        thorax_width_mm = 0
        if thorax_props:
            minr_t, minc_t, maxr_t, maxc_t = thorax_props[0].bbox
            thorax_width_mm = (maxc_t - minc_t) * meta["pixel_spacing_x_mm"]
            
        actr = cardiac_width_mm / thorax_width_mm if thorax_width_mm > 0 else 0
        
        # 2. RADIOMICS / DENSITY (HOUNSFIELD UNITS)
        # Extract the true physical density of the heart muscle
        heart_hu_pixels = hu_image[heart_mask]
        mean_hu = np.mean(heart_hu_pixels) if len(heart_hu_pixels) > 0 else 0
        var_hu = np.var(heart_hu_pixels) if len(heart_hu_pixels) > 0 else 0
        
        # 3. TEXTURE (GLCM)
        # Normalizing HU for GLCM computation (must be positive integers)
        bbox_hu = hu_image[minr_h:maxr_h, minc_h:maxc_h]
        bbox_norm = exposure.rescale_intensity(bbox_hu, out_range=(0, 255)).astype(np.uint8)
        
        glcm_contrast = 0; glcm_homog = 0
        if bbox_norm.size > 0:
            glcm = graycomatrix(bbox_norm, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
            glcm_homog = graycoprops(glcm, 'homogeneity')[0, 0]

        return {
            "Cardiac Area (cm^2)": round(cardiac_area_cm2, 2),
            "Thoracic Area (cm^2)": round(thorax_area_cm2, 2),
            "CTAR (Area Ratio)": round(ctar, 4),
            "Max Cardiac Width (mm)": round(cardiac_width_mm, 2),
            "aCTR (Axial Width Ratio)": round(actr, 4),
            "Mean Density (HU)": round(mean_hu, 2),
            "Density Variance (HU Variance)": round(var_hu, 2),
            "GLCM Contrast": round(glcm_contrast, 4),
            "GLCM Homogeneity": round(glcm_homog, 4)
        }

    def step_process_image(self, hu_image, meta, title):
        """
        GUI ADAPTATION:
        Execute CT Pipeline: Windowing -> Multi-Otsu -> Morphology -> Extraction.
        Returns Pandas DataFrame and Figure objects.
        """
        # 1. CT WINDOWING (Enhancement Specific for CT)
        windowed_img = self.apply_ct_windowing(hu_image, window_level=40, window_width=400)
        
        # 2. MULTI-OTSU THRESHOLDING
        # Separate into 3 Classes: Lungs/Air(0), Soft Tissue/Heart(1), Bones(2)
        thresholds = filters.threshold_multiotsu(windowed_img, classes=3)
        regions = np.digitize(windowed_img, bins=thresholds)
        
        soft_tissue_mask = (regions == 1)
        bone_mask = (regions == 2)
        
        # Thorax Mask (Soft tissue + Bone) filled to estimate chest cavity
        thorax_mask = ndi.binary_fill_holes(soft_tissue_mask | bone_mask)
        
        # 3. MORPHOLOGICAL SCALPEL (Isolate Heart)
        # Opening to cut the heart off from the spine, sternum, and chest wall
        opened_mask = morphology.binary_opening(soft_tissue_mask, morphology.disk(5))
        
        # Extract the largest blob (The Heart)
        labeled_mask = measure.label(opened_mask)
        if labeled_mask.max() > 0:
            largest_blob = max(measure.regionprops(labeled_mask), key=lambda x: x.area)
            heart_core = (labeled_mask == largest_blob.label)
        else:
            heart_core = opened_mask
            
        # Smooth the cardiac silhouette
        heart_filled = ndi.binary_fill_holes(heart_core)
        heart_silhouette = morphology.binary_closing(heart_filled, morphology.disk(5))
        
        silhouette_edge = morphology.binary_dilation(heart_silhouette, morphology.disk(1)) ^ heart_silhouette

        # 4. FEATURE EXTRACTION and EXPORT
        feat_dict = self.extract_features(heart_silhouette, hu_image, thorax_mask, meta)
        
        descriptions = {
            "Cardiac Area (cm^2)": "Physical area of the heart slice (using PixelSpacing).",
            "Thoracic Area (cm^2)": "Physical area of the inner chest cavity.",
            "CTAR (Area Ratio)": "Cardiothoracic Area Ratio (Cardiac Area / Thoracic Area).",
            "Max Cardiac Width (mm)": "Maximum transverse diameter of the heart.",
            "aCTR (Axial Width Ratio)": "Axial Cardiothoracic Ratio. >0.5 indicates Cardiomegaly.",
            "Mean Density (HU)": "Average Hounsfield Unit of myocardium (Normal: ~40 HU non-contrast).",
            "Density Variance (HU Variance)": "Heterogeneity of heart tissue (Detects calcifications).",
            "GLCM Contrast": "Radiomics texture feature (Myocardial fibrosis indicator).",
            "GLCM Homogeneity": "Radiomics texture smoothness."
        }

        features_data = [{"CT Radiomics Feature": key, "Description": descriptions[key], "Value": val} 
                         for key, val in feat_dict.items()]
        df_features = pd.DataFrame(features_data)
        
        export_file = os.path.join(self.export_dir, f"CT_Features_{title}.xlsx")
        df_features.to_excel(export_file, index=False)
        print(f"[SUCCESS] Feature data exported to: {export_file}")

        # 5. VISUALIZATION PLOTS
        # PLOT 1: Windowing and Multi-Otsu
        fig_otsu, ax_otsu = plt.subplots(1, 3, figsize=(18, 5))
        
        # Displaying raw HU image can be tricky, so we show the windowed version
        ax_otsu[0].imshow(windowed_img, cmap='bone'); ax_otsu[0].set_title("1. Mediastinal Window (WL=40, WW=400)"); ax_otsu[0].grid(False)
        ax_otsu[1].imshow(regions, cmap='viridis'); ax_otsu[1].set_title("2. Multi-Otsu (Air, Soft Tissue, Bone)"); ax_otsu[1].grid(False)
        ax_otsu[2].imshow(soft_tissue_mask, cmap='gray'); ax_otsu[2].set_title("3. Class 1 Extracted (Soft Tissue)"); ax_otsu[2].grid(False)
        fig_otsu.tight_layout()

        # PLOT 2: Morphological Processing
        fig_prog, ax_prog = plt.subplots(1, 3, figsize=(18, 6))
        ax_prog[0].imshow(opened_mask, cmap='gray'); ax_prog[0].set_title("4. Opening (Detaching Spine/Sternum)"); ax_prog[0].grid(False)
        ax_prog[1].imshow(heart_silhouette, cmap='gray'); ax_prog[1].set_title("5. Largest Blob Isolated and Filled"); ax_prog[1].grid(False)
        
        # Overlay on windowed image
        overlay = np.stack((windowed_img,)*3, axis=-1)
        overlay[silhouette_edge] = [255, 0, 0] # Red Edge
        
        ax_prog[2].imshow(overlay); ax_prog[2].set_title("6. Cardiac Edge Overlay"); ax_prog[2].grid(False)
        fig_prog.tight_layout()

        # PLOT 3: Final Output and Extracted ROI
        filled_white_mask = np.zeros_like(windowed_img, dtype=np.uint8)
        filled_white_mask[heart_silhouette] = 255
        
        extracted_heart = np.zeros_like(windowed_img)
        extracted_heart[heart_silhouette] = windowed_img[heart_silhouette]

        fig_final, ax_final = plt.subplots(1, 3, figsize=(18, 6))
        
        ax_final[0].imshow(overlay)
        ax_final[0].set_title("Final Cardiac Silhouette")
        ax_final[0].grid(False)
        
        if labeled_mask.max() > 0:
            props = measure.regionprops(measure.label(heart_silhouette))[0]
            y0, x0 = props.centroid
            ax_final[0].plot(x0, y0, marker='o', color='yellow', markersize=6, label='Centroid')
            
            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax_final[0].plot(bx, by, '-c', linewidth=2, label='Bounding Box')
            ax_final[0].legend(loc='upper right', fontsize=8)

        ax_final[1].imshow(filled_white_mask, cmap='gray')
        ax_final[1].set_title("Solid Filled ROI Mask")
        ax_final[1].grid(False)
        
        ax_final[2].imshow(extracted_heart, cmap='bone')
        ax_final[2].set_title("Extracted Heart (Windowed)")
        ax_final[2].grid(False)
        
        fig_final.tight_layout()

        # RETURN figures and dataframe for GUI insertion
        return df_features, fig_otsu, fig_prog, fig_final

# Note: The CLI testing block (__main__) is removed since this is now a GUI library component