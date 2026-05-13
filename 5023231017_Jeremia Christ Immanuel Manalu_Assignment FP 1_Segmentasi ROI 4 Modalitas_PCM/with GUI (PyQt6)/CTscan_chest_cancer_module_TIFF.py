'''
Final Project 1 
- Using CT image (TIFF file format) for heart segmentation from Thorax/Chest section
- GUI-Ready Module: Refactored to return Matplotlib Figures and Pandas DataFrames instead of blocking with plt.show() or input()
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v3 as imageio
import scipy.ndimage as ndi
from skimage import exposure, measure, morphology, filters
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity
from skimage.measure import shannon_entropy
import math

# Use white style for a clean background
sns.set_theme(style="white")

def view_all_tiffs(tiff_dir):
    """
    Reads all TIFF files, sorts them anatomically/alphabetically,
    and opens an Interactive Slice Viewer for user selection.
    """
    print(f"\n[INFO] Scanning directory: {tiff_dir}")
    files = sorted([f for f in os.listdir(tiff_dir) if f.lower().endswith(('.tif', '.tiff'))])
    
    if not files:
        print("[ERROR] No TIFF files found in the specified directory.")
        return []

    print(f"[INFO] Found {len(files)} TIFF files. Loading...")
    
    images = []
    valid_tiffs = []
    for f in files:
        path = os.path.join(tiff_dir, f)
        try:
            img = imageio.imread(path)
            # Convert to grayscale if it happens to be RGB
            if img.ndim == 3:
                img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            images.append(img)
            valid_tiffs.append(f)
        except Exception:
            continue

    if not images:
        print("[ERROR] Could not read any valid TIFF pixel data.")
        return []

    # INTERACTIVE SCROLLABLE MATPLOTLIB VIEWER
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("INTERACTIVE TIFF VIEWER, CLOSE TO PROCEED")
    
    ax.volume = images
    ax.index = len(images) // 2 
    ax.filenames = valid_tiffs

    img_display = ax.imshow(ax.volume[ax.index], cmap='bone')
    ax.set_title(f"Index [{ax.index}] | File: {ax.filenames[ax.index]}\nScroll Mouse UP/DOWN or Use ARROW KEYS", fontsize=12, fontweight='bold')
    ax.axis('off')

    def update_slice():
        img_display.set_array(ax.volume[ax.index])
        ax.set_title(f"Index [{ax.index}] | File: {ax.filenames[ax.index]}\nScroll Mouse UP/DOWN or Use ARROW KEYS", fontsize=12, fontweight='bold')
        fig.canvas.draw()

    def process_scroll(event):
        if event.button == 'up':
            ax.index = (ax.index + 1) % len(ax.volume)
        elif event.button == 'down':
            ax.index = (ax.index - 1) % len(ax.volume)
        update_slice()

    def process_key(event):
        if event.key in ['right', 'up']:
            ax.index = (ax.index + 1) % len(ax.volume)
        elif event.key in ['left', 'down']:
            ax.index = (ax.index - 1) % len(ax.volume)
        update_slice()

    fig.canvas.mpl_connect('scroll_event', process_scroll)
    fig.canvas.mpl_connect('key_press_event', process_key)

    print("[INFO] Displaying Interactive TIFF Viewer.")
    print("[INFO] Scroll to find the best 2 slices, note their Index numbers.")
    print("[INFO] CLOSE THE PLOT WINDOW TO CONTINUE...")
    
    plt.tight_layout()
    plt.show()
    
    return valid_tiffs

class CTTiffProcessor:
    def __init__(self, tiff_dir, selected_files, crop_coords, export_dir):
        # Initialize CT dataset with selected TIFF files and crop coordinates
        self.dataset = {}
        for f in selected_files:
            path = os.path.join(tiff_dir, f)
            img = imageio.imread(path)
            if img.ndim == 3:
                img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
            # Normalize to 0-255 if it's a 16-bit TIFF
            if img.dtype != np.uint8:
                img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
                
            self.dataset[f] = img
            
        self.crop_coords = crop_coords
        self.export_dir = export_dir
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def calc_metrics(self, img_orig, img_enh):
        orig_norm = img_orig.astype(np.float64)
        enh_norm = img_enh.astype(np.float64)
        rmse = np.sqrt(mean_squared_error(orig_norm, enh_norm))
        psnr = peak_signal_noise_ratio(orig_norm, enh_norm, data_range=255)
        ssim = structural_similarity(img_orig, img_enh, data_range=255)
        entropy = shannon_entropy(img_enh)
        return round(rmse, 2), round(psnr, 2), round(ssim, 4), round(entropy, 4)

    def plot_hist_with_ogive(self, data, ax, color, title):
        # Helper Function: Plot Histogram + Ogive (CDF), filtering out pure black <= 15
        filtered_data = data[data > 15]
        
        if len(filtered_data) == 0:
            ax.set_title(title + " (No Data > 15)")
            return

        sns.histplot(filtered_data, bins=100, binrange=(16, 255), color=color, ax=ax, element='step')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        counts, bin_edges = np.histogram(filtered_data, bins=100, range=(16, 255))
        cdf = np.cumsum(counts) / np.sum(counts) 
        cdf = np.insert(cdf, 0, 0) 
        
        ax_ogive = ax.twinx()
        ax_ogive.plot(bin_edges, cdf, color='black', linewidth=1.5, linestyle='--')
        ax_ogive.set_ylabel('Cumulative Prob. (0-1)')
        ax_ogive.set_ylim(0, 1.05)
        ax_ogive.grid(False)

    def step1_plot_initial_and_crop(self, img_gray, title):
        """
        GUI ADAPTATION:
        Plot initial image, apply cropping, and return cropped image along with figure.
        """
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
        axes1[0].imshow(img_gray, cmap='bone')
        axes1[0].set_title("Original CT Slice (Check crop coordinates here)")
        axes1[0].grid(False)
        self.plot_hist_with_ogive(img_gray.ravel(), axes1[1], 'gray', "Original Image Histogram")
        fig1.tight_layout()

        x_min, x_max, y_min, y_max = self.crop_coords.get(title, (0, img_gray.shape[1], 0, img_gray.shape[0]))
        x_min, x_max = max(0, x_min), min(img_gray.shape[1], x_max)
        y_min, y_max = max(0, y_min), min(img_gray.shape[0], y_max)
        
        img_crop = img_gray[y_min:y_max, x_min:x_max]
        print(f"-> Cropped at X:({x_min}-{x_max}), Y:({y_min}-{y_max})")
        
        return img_crop, (x_min, x_max, y_min, y_max), fig1

    def step2_evaluate_enhancement(self, img_crop, title):
        """
        GUI ADAPTATION:
        Evaluate performance, Bar Charts.
        REMOVED terminal input(). Returns a dictionary of images so GUI can choose later.
        """
        p2, p98 = np.percentile(img_crop, (2, 98))
        img_cs = exposure.rescale_intensity(img_crop, in_range=(p2, p98), out_range=(0, 255)).astype(np.uint8)
        img_he = (exposure.equalize_hist(img_crop) * 255).astype(np.uint8)
        img_clahe = (exposure.equalize_adapthist(img_crop, clip_limit=0.03) * 255).astype(np.uint8)

        methods = [("Contrast Stretching", img_cs), ("Histogram Eq", img_he), ("CLAHE", img_clahe)]
        metrics_data = []
        for name, img in methods:
            rmse, psnr, ssim, entropy = self.calc_metrics(img_crop, img)
            metrics_data.append({"Enhancement Method": name, "RMSE": rmse, "PSNR (dB)": psnr, "SSIM": ssim, "Entropy": entropy})
            
        df_metrics = pd.DataFrame(metrics_data)
        
        fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
        images = [img_crop, img_cs, img_he, img_clahe]
        titles = ["Original Cropped", "Contrast Stretching", "Histogram Eq", "CLAHE"]
        
        for i in range(4):
            axes2[0, i].imshow(images[i], cmap='bone'); axes2[0, i].set_title(titles[i], fontsize=12); axes2[0, i].axis('off')
            self.plot_hist_with_ogive(images[i].ravel(), axes2[1, i], 'teal', f"Hist+Ogive: {titles[i]}")
        fig2.tight_layout()

        fig_bar, ax_bar = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.barplot(data=df_metrics, x="Enhancement Method", y="RMSE", hue="Enhancement Method", legend=False, ax=ax_bar[0,0], palette="Reds")
        ax_bar[0,0].set_title("RMSE (Lower is Better)")
        sns.barplot(data=df_metrics, x="Enhancement Method", y="PSNR (dB)", hue="Enhancement Method", legend=False, ax=ax_bar[0,1], palette="Greens")
        ax_bar[0,1].set_title("PSNR (Higher is Better)")
        sns.barplot(data=df_metrics, x="Enhancement Method", y="SSIM", hue="Enhancement Method", legend=False, ax=ax_bar[1,0], palette="Blues")
        ax_bar[1,0].set_title("SSIM (Preserves structures)")
        sns.barplot(data=df_metrics, x="Enhancement Method", y="Entropy", hue="Enhancement Method", legend=False, ax=ax_bar[1,1], palette="Purples")
        ax_bar[1,1].set_title("Entropy (Texture Detail)")
        fig_bar.tight_layout()

        enhanced_images_dict = {
            "Contrast Stretching": img_cs,
            "Histogram Equalization": img_he,
            "CLAHE": img_clahe
        }

        return enhanced_images_dict, df_metrics, fig2, fig_bar

    def extract_features(self, heart_mask, img_crop):
        # Extract Morphological and Texture features using Pixel units
        labeled_heart = measure.label(heart_mask)
        props = measure.regionprops(labeled_heart, intensity_image=img_crop)
        
        if not props:
            return {}
        
        prop = props[0]
        
        cardiac_area_px = prop.area
        minr_h, minc_h, maxr_h, maxc_h = prop.bbox
        cardiac_width_px = maxc_h - minc_h
        
        # INTENSITY (0-255)
        heart_pixels = img_crop[heart_mask]
        mean_int = np.mean(heart_pixels) if len(heart_pixels) > 0 else 0
        var_int = np.var(heart_pixels) if len(heart_pixels) > 0 else 0
        
        # TEXTURE (GLCM)
        bbox_img = img_crop[minr_h:maxr_h, minc_h:maxc_h]
        glcm_contrast = 0; glcm_homog = 0
        if bbox_img.size > 0:
            glcm = graycomatrix(bbox_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
            glcm_homog = graycoprops(glcm, 'homogeneity')[0, 0]

        return {
            "Cardiac Area (px^2)": round(cardiac_area_px, 2),
            "Max Cardiac Width (px)": round(cardiac_width_px, 2),
            "Mean Intensity (0-255)": round(mean_int, 2),
            "Intensity Variance": round(var_int, 2),
            "GLCM Contrast": round(glcm_contrast, 4),
            "GLCM Homogeneity": round(glcm_homog, 4)
        }

    def step3_segment_and_extract(self, best_img, img_crop, img_full, coords, title, method_name):
        """
        GUI ADAPTATION:
        Standard Otsu Segmentation, Morphology, and Extraction on Cropped ROI.
        Returns Pandas DataFrame and Figure objects.
        """
        x_min, x_max, y_min, y_max = coords
        
        # 1. LOW-PASS FILTER (GAUSSIAN BLUR)
        blurred_img = filters.gaussian(best_img, sigma=2, preserve_range=True)
        
        # 2. STANDARD OTSU THRESHOLDING (2 CLASSES)
        thresh_val = filters.threshold_otsu(blurred_img)
        binary_mask = (blurred_img > thresh_val)
        
        # 3. MORPHOLOGICAL SCALPEL & BLOB EXTRACTION
        cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=200)
        
        labeled_mask = measure.label(cleaned_mask)
        if labeled_mask.max() > 0:
            largest_blob = max(measure.regionprops(labeled_mask), key=lambda x: x.area)
            heart_core = (labeled_mask == largest_blob.label)
        else:
            heart_core = cleaned_mask
            
        heart_filled = ndi.binary_fill_holes(heart_core)
        heart_silhouette = morphology.binary_closing(heart_filled, morphology.disk(5))
        
        silhouette_edge = morphology.binary_dilation(heart_silhouette, morphology.disk(1)) ^ heart_silhouette

        # 4. FEATURE EXTRACTION & EXPORT
        feat_dict = self.extract_features(heart_silhouette, img_crop)
        
        descriptions = {
            "Cardiac Area (px^2)": "Pixel area of the isolated heart silhouette.",
            "Max Cardiac Width (px)": "Maximum transverse diameter in pixels.",
            "Mean Intensity (0-255)": "Average pixel brightness of myocardium.",
            "Intensity Variance": "Heterogeneity of heart tissue (e.g., detecting calcification).",
            "GLCM Contrast": "Texture feature (Myocardial fibrosis indicator).",
            "GLCM Homogeneity": "Texture smoothness."
        }

        features_data = [{"Feature": key, "Description": descriptions[key], "Value": val} for key, val in feat_dict.items()]
        df_features = pd.DataFrame(features_data)
        
        export_file = os.path.join(self.export_dir, f"CT_TIFF_Features_{title}.xlsx")
        df_features.to_excel(export_file, index=False)
        print(f"[SUCCESS] Feature data exported to: {export_file}")

        # 5. PLOTTING VISUALIZATIONS
        fig_otsu, ax_otsu = plt.subplots(1, 3, figsize=(18, 5))
        ax_otsu[0].imshow(best_img, cmap='bone'); ax_otsu[0].set_title("1. Enhanced Cropped CT"); ax_otsu[0].axis('off')
        ax_otsu[1].imshow(blurred_img, cmap='bone'); ax_otsu[1].set_title("2. Gaussian Blur"); ax_otsu[1].axis('off')
        ax_otsu[2].imshow(binary_mask, cmap='gray'); ax_otsu[2].set_title("3. Standard Otsu Mask"); ax_otsu[2].axis('off')
        fig_otsu.tight_layout()

        fig_prog, ax_prog = plt.subplots(1, 3, figsize=(18, 6))
        ax_prog[0].imshow(cleaned_mask, cmap='gray'); ax_prog[0].set_title("4. Noise Removed"); ax_prog[0].axis('off')
        ax_prog[1].imshow(heart_silhouette, cmap='gray'); ax_prog[1].set_title("5. Largest Blob & Smoothed"); ax_prog[1].axis('off')
        
        overlay_crop = np.stack((img_crop,)*3, axis=-1)
        overlay_crop[silhouette_edge] = [255, 0, 0] # Red Edge
        ax_prog[2].imshow(overlay_crop); ax_prog[2].set_title("6. Cardiac Edge Overlay (Cropped)"); ax_prog[2].axis('off')
        fig_prog.tight_layout()

        # FINAL FULL-SIZE MAPPING
        filled_white_mask = np.zeros_like(img_crop, dtype=np.uint8)
        filled_white_mask[heart_silhouette] = 255
        
        extracted_heart = np.zeros_like(img_crop)
        extracted_heart[heart_silhouette] = img_crop[heart_silhouette]
        
        full_edge = np.zeros_like(img_full, dtype=bool)
        full_edge[y_min:y_max, x_min:x_max] = silhouette_edge
        overlay_full = np.stack((img_full,)*3, axis=-1)
        overlay_full[full_edge] = [255, 0, 0]

        fig_final, ax_final = plt.subplots(1, 3, figsize=(18, 6))
        ax_final[0].imshow(overlay_full)
        ax_final[0].set_title("Cardiac Silhouette on Full CT Slice")
        ax_final[0].axis('off')
        
        if labeled_mask.max() > 0:
            props = measure.regionprops(measure.label(heart_silhouette))[0]
            y0_c, x0_c = props.centroid
            y0_full, x0_full = y0_c + y_min, x0_c + x_min
            ax_final[0].plot(x0_full, y0_full, marker='o', color='yellow', markersize=6, label='Centroid')
            
            minr, minc, maxr, maxc = props.bbox
            bx = (minc+x_min, maxc+x_min, maxc+x_min, minc+x_min, minc+x_min)
            by = (minr+y_min, minr+y_min, maxr+y_min, maxr+y_min, minr+y_min)
            ax_final[0].plot(bx, by, '-c', linewidth=2, label='Bounding Box')
            ax_final[0].legend(loc='upper right', fontsize=8)

        ax_final[1].imshow(filled_white_mask, cmap='gray')
        ax_final[1].set_title("Solid Filled ROI Mask")
        ax_final[1].axis('off')
        
        ax_final[2].imshow(extracted_heart, cmap='bone')
        ax_final[2].set_title("Extracted Heart")
        ax_final[2].axis('off')
        fig_final.tight_layout()

        # RETURN figures and dataframe for GUI insertion
        return df_features, fig_otsu, fig_prog, fig_final

# Note: The CLI testing block (__main__) is removed since this is now a GUI library component