'''
Final Project 1 
- X-Ray images for lung segmentation, 2 Pneumonia and 2 Normal (JPEG file format) 
- GUI-Ready Module: Refactored to return Matplotlib Figures and Pandas DataFrames instead of blocking with plt.show() or input()
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v3 as imageio
import scipy.ndimage as ndi
from scipy.stats import skew, kurtosis
from skimage import exposure, measure, morphology, filters
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity
import math

# Use white style for a clean background
sns.set_theme(style="white")

class XRayProcessor:
    def __init__(self, image_paths, crop_coords, export_dir):
        # Initialize X-Ray dataset, crop coordinates, and export directory
        self.dataset = {}
        for title, path in image_paths.items():
            img = imageio.imread(path)
            # Convert to grayscale if RGB
            if img.ndim == 3:
                img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            self.dataset[title] = img
            
        self.crop_coords = crop_coords
        self.export_dir = export_dir

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def calc_metrics(self, img_orig, img_enh):
        """Calculate RMSE, PSNR, SSIM, and Shannon Entropy metrics."""
        orig_norm = img_orig.astype(np.float64)
        enh_norm = img_enh.astype(np.float64)
        
        rmse = np.sqrt(mean_squared_error(orig_norm, enh_norm))
        psnr = peak_signal_noise_ratio(orig_norm, enh_norm, data_range=255)
        ssim = structural_similarity(img_orig, img_enh, data_range=255)
        entropy = shannon_entropy(img_enh)
        
        return round(rmse, 2), round(psnr, 2), round(ssim, 4), round(entropy, 4)

    def plot_hist_with_ogive(self, data, ax, color, title):
        # Plot Histogram + Ogive (CDF), filtering out pure black background <= 15
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

    def step1_plot_initial(self, img_gray, title):
        """
        GUI ADAPTATION:
        Display Original Grayscale X-Ray Image and Histogram.
        Returns the Matplotlib Figure object.
        """
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
        
        axes1[0].imshow(img_gray, cmap='gray')
        axes1[0].set_title("Original CXR (Check crop coordinates here)")
        axes1[0].grid(False)
        
        self.plot_hist_with_ogive(img_gray.ravel(), axes1[1], 'gray', "Original Image Histogram")
        fig1.tight_layout()
        
        return fig1

    def step2_crop(self, img_gray, title):
        # 1. Use Otsu to find dark areas (lung air and external background)
        lung_thresh = filters.threshold_otsu(img_gray)
        lung_mask = img_gray < lung_thresh 
        
        # 2. Label all dark regions
        labeled_lungs = measure.label(lung_mask)
        lung_props = measure.regionprops(labeled_lungs)
        
        valid_lungs = []
        height, width = img_gray.shape
        
        for p in lung_props:
            # Filter 1: Remove small noise (dynamic: minimum 0.5% of total image area)
            if p.area < (height * width * 0.005):
                continue
                
            y_c, x_c = p.centroid
            
            # Filter 2: Remove external background at the left/right edges based on Centroid.
            # The centroid of the patient's lungs is always located more towards the center of the image.
            # If the centroid is very close to the outer edge (< 8% or > 92% of the width),
            # then it is definitely external background.
            if x_c < width * 0.08 or x_c > width * 0.92:
                continue
                
            # Save the valid lung objects
            valid_lungs.append(p)
        
        left_bound = 0
        right_bound = width
        lung_mask_cleaned = np.zeros_like(lung_mask)
        
        if valid_lungs:
            # 3. Take the 2 largest valid lung regions (in case of some fragmentation) and create a clean mask
            lung_props_sorted = sorted(valid_lungs, key=lambda x: x.area, reverse=True)
            top_lungs = lung_props_sorted[:2]
            
            # Create a clean mask containing only the lungs to visualize
            for p in top_lungs:
                lung_mask_cleaned[labeled_lungs == p.label] = True
            
            # Find the leftmost and rightmost bounds of the lungs to estimate thoracic width
            min_cols = [p.bbox[1] for p in top_lungs]
            max_cols = [p.bbox[3] for p in top_lungs]
            left_bound = min(min_cols)
            right_bound = max(max_cols)
            thoracic_width = right_bound - left_bound
        else:
            thoracic_width = width # Fallback
            
        print(f"[INFO] Estimasi Thoracic Width pada citra asli: {thoracic_width} px")
        
        # Crop the image manually to isolate the lower-middle chest (Heart and Lungs region)
        x_min, x_max, y_min, y_max = self.crop_coords.get(title, (0, width, 0, height))
        x_min, x_max = max(0, x_min), min(width, x_max)
        y_min, y_max = max(0, y_min), min(height, y_max)
        
        fig_lung, ax_lung = plt.subplots(1, 2, figsize=(14, 5))
        ax_lung[0].imshow(lung_mask_cleaned, cmap='gray')
        ax_lung[0].set_title("Cleaned Lung Mask (Centroid Filtered)")
        ax_lung[0].grid(False)
        
        ax_lung[1].imshow(img_gray, cmap='gray')
        ax_lung[1].axvline(x=left_bound, color='cyan', linestyle='--', linewidth=2, label='Thoracic Bounds')
        ax_lung[1].axvline(x=right_bound, color='cyan', linestyle='--', linewidth=2)
        
        bx = (x_min, x_max, x_max, x_min, x_min)
        by = (y_min, y_min, y_max, y_max, y_min)
        ax_lung[1].plot(bx, by, '-g', linewidth=2, label='Crop ROI Reference')
        ax_lung[1].set_title(f"Thoracic Width Estimation: {thoracic_width} px")
        ax_lung[1].legend(loc='upper right', fontsize=10)
        ax_lung[1].grid(False)
        fig_lung.tight_layout()
        
        img_crop = img_gray[y_min:y_max, x_min:x_max]
        print(f"-> Cropped at X:({x_min}-{x_max}), Y:({y_min}-{y_max})")
        
        return img_crop, (x_min, x_max, y_min, y_max), thoracic_width, fig_lung

    def step3_evaluate_enhancement(self, img_crop, title):
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
            metrics_data.append({
                "Enhancement Method": name, 
                "RMSE (levels)": rmse, 
                "PSNR (dB)": psnr, 
                "SSIM": ssim, 
                "Entropy (bits)": entropy
            })
            
        df_metrics = pd.DataFrame(metrics_data)
        
        fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
        images = [img_crop, img_cs, img_he, img_clahe]
        titles = ["Original Cropped", "Contrast Stretching", "Histogram Eq", "CLAHE"]
        
        for i in range(4):
            axes2[0, i].imshow(images[i], cmap='gray'); axes2[0, i].set_title(titles[i], fontsize=12); axes2[0, i].grid(False) 
            self.plot_hist_with_ogive(images[i].ravel(), axes2[1, i], 'teal', f"Hist+Ogive: {titles[i]}")
        fig2.tight_layout()

        fig_bar, ax_bar = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.barplot(data=df_metrics, x="Enhancement Method", y="RMSE (levels)", hue="Enhancement Method", legend=False, ax=ax_bar[0,0], palette="Reds")
        ax_bar[0,0].set_title("RMSE (Lower is Better)")
        sns.barplot(data=df_metrics, x="Enhancement Method", y="PSNR (dB)", hue="Enhancement Method", legend=False, ax=ax_bar[0,1], palette="Greens")
        ax_bar[0,1].set_title("PSNR (Higher is Better)")
        sns.barplot(data=df_metrics, x="Enhancement Method", y="SSIM", hue="Enhancement Method", legend=False, ax=ax_bar[1,0], palette="Blues")
        ax_bar[1,0].set_title("SSIM (Preserves structures)")
        sns.barplot(data=df_metrics, x="Enhancement Method", y="Entropy (bits)", hue="Enhancement Method", legend=False, ax=ax_bar[1,1], palette="Purples")
        ax_bar[1,1].set_title("Entropy (Vascular and Infection Detail)")
        fig_bar.tight_layout()

        # Dictionary to store the enhanced images so the GUI can pick one based on Combobox selection
        enhanced_images_dict = {
            "Contrast Stretching": img_cs,
            "Histogram Equalization": img_he,
            "CLAHE": img_clahe
        }

        return enhanced_images_dict, df_metrics, fig2, fig_bar

    def extract_features(self, heart_mask, img_crop, thoracic_width):
        # Extract Morphological and Texture features, including Edge Gradient Strength and CTR.
        labeled_heart = measure.label(heart_mask)
        props = measure.regionprops(labeled_heart, intensity_image=img_crop)
        
        if not props:
            return {k: 0 for k in ['Area (px^2)', 'Perimeter (px)', 'Cardiac Width (px)', 'Thoracic Width (px)', 'CTR (Ratio)', 'Circularity', 
                                   'Mean Intensity (0-255)', 'Intensity Variance', 'GLCM Contrast', 
                                   'GLCM Homogeneity', 'Edge Gradient Strength']}
        
        prop = props[0]
        area = prop.area
        perimeter = prop.perimeter
        
        # 1. CARDIAC WIDTH
        minr, minc, maxr, maxc = prop.bbox
        cardiac_width = maxc - minc
            
        # 2. CARDIOTHORACIC RATIO (CTR)
        ctr = cardiac_width / thoracic_width if thoracic_width > 0 else 0
        
        circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        heart_pixels = img_crop[heart_mask]
        mean_int = prop.mean_intensity
        var_int = np.var(heart_pixels) if len(heart_pixels) > 0 else 0
        
        # Texture
        bbox_img = img_crop[minr:maxr, minc:maxc]
        glcm_contrast = 0; glcm_homog = 0
        if bbox_img.size > 0:
            glcm = graycomatrix(bbox_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
            glcm_homog = graycoprops(glcm, 'homogeneity')[0, 0]
            
        # Edge Strength
        sobel_grad = filters.sobel(img_crop)
        edge_mask = morphology.binary_dilation(heart_mask, morphology.disk(1)) ^ heart_mask
        edge_pixels = sobel_grad[edge_mask]
        edge_strength = np.mean(edge_pixels) if len(edge_pixels) > 0 else 0

        return {
            "Area (px^2)": round(area, 2),
            "Perimeter (px)": round(perimeter, 2),
            "Cardiac Width (px)": cardiac_width,
            "Thoracic Width (px)": thoracic_width,
            "CTR (Ratio)": round(ctr, 4),
            "Circularity": round(circularity, 4),
            "Mean Intensity (0-255)": round(mean_int, 2),
            "Intensity Variance": round(var_int, 2),
            "GLCM Contrast": round(glcm_contrast, 4),
            "GLCM Homogeneity": round(glcm_homog, 4),
            "Edge Gradient Strength": round(edge_strength, 4)
        }

    def step4_segment_and_extract(self, best_img, img_crop, img_full, coords, thoracic_width, title, method_name):
        """
        GUI ADAPTATION:
        Low-Pass Filtering Segmentation, Organic Masking, Overlays, and Features.
        Returns Pandas DataFrame and Figure objects.
        """
        from skimage.segmentation import clear_border 
        
        x_min, x_max, y_min, y_max = coords
        
        # 1. LOW-PASS FILTERING (HEAVY GAUSSIAN BLUR)
        blurred_img = filters.gaussian(best_img, sigma=5, preserve_range=True)
        
        # 2. ADAPTIVE THRESHOLDING
        try:
            thresholds = filters.threshold_multiotsu(blurred_img, classes=3)
            regions = np.digitize(blurred_img, bins=thresholds)
            dense_mask = (regions >= 1)
        except ValueError:
            thresh_val = filters.threshold_otsu(blurred_img)
            dense_mask = (blurred_img > thresh_val)
        
        # 3. MORPHOLOGICAL OPENING
        opened_mask = morphology.binary_opening(dense_mask, morphology.disk(25))
        
        # 4. LARGEST BLOB EXTRACTION
        labeled_mask = measure.label(opened_mask)
        if labeled_mask.max() > 0:
            largest_blob = max(measure.regionprops(labeled_mask), key=lambda x: x.area)
            heart_core = (labeled_mask == largest_blob.label)
        else:
            heart_core = opened_mask
            
        # 5. ORGANIC EDGE REFINEMENT and PADDED CLOSED BORDER
        heart_filled = ndi.binary_fill_holes(heart_core)
        heart_silhouette = morphology.binary_closing(heart_filled, morphology.disk(15))
        
        padded_mask = np.pad(heart_silhouette, pad_width=3, mode='constant', constant_values=0)
        padded_edge = morphology.binary_dilation(padded_mask, morphology.disk(2)) ^ padded_mask
        silhouette_edge = padded_edge[3:-3, 3:-3] 

        # 6. FEATURE EXTRACTION and EXPORT
        feat_dict = self.extract_features(heart_silhouette, img_crop, thoracic_width)
        
        descriptions = {
            "Area (px^2)": "Total area of the cardiac silhouette.",
            "Perimeter (px)": "Circumference of the heart shape.",
            "Cardiac Width (px)": "Maximum horizontal span of the heart silhouette.",
            "Thoracic Width (px)": "Maximum horizontal span of the inner ribcage/lungs.",
            "CTR (Ratio)": "Cardiothoracic Ratio (Cardiac Width / Thoracic Width). >0.5 indicates Cardiomegaly.",
            "Circularity": "Roundness of the cardiac silhouette.",
            "Mean Intensity (0-255)": "Average density (Useful to detect fluid overlap).",
            "Intensity Variance": "Homogeneity of the heart tissue (Higher in severe pneumonia).",
            "GLCM Contrast": "Local texture variation.",
            "GLCM Homogeneity": "Texture smoothness inside the silhouette.",
            "Edge Gradient Strength": "Sharpness of the heart border (Low = Silhouette Sign / Pneumonia)."
        }

        features_data = [{"Cardiothoracic Feature": key, "Description": descriptions[key], "Value": val} 
                         for key, val in feat_dict.items()]
        df_features = pd.DataFrame(features_data)
        
        export_file = os.path.join(self.export_dir, f"XRay_Features_{title}.xlsx")
        df_features.to_excel(export_file, index=False)
        print(f"[SUCCESS] Feature data successfully exported to: {export_file}")

        # 7. PLOTTING VISUALIZATIONS
        fig_otsu, ax_otsu = plt.subplots(1, 3, figsize=(18, 5))
        ax_otsu[0].imshow(best_img, cmap='gray'); ax_otsu[0].set_title("1. Enhanced CXR"); ax_otsu[0].grid(False)
        ax_otsu[1].imshow(blurred_img, cmap='gray'); ax_otsu[1].set_title("2. Gaussian Blur (sigma=5)"); ax_otsu[1].grid(False)
        ax_otsu[2].imshow(dense_mask, cmap='gray'); ax_otsu[2].set_title("3. Thresholded Dense Mask"); ax_otsu[2].grid(False)
        fig_otsu.tight_layout()

        fig_prog, ax_prog = plt.subplots(1, 3, figsize=(18, 6))
        ax_prog[0].imshow(opened_mask, cmap='gray'); ax_prog[0].set_title("4. Morphological Opening (Cut Connections)"); ax_prog[0].grid(False)
        ax_prog[1].imshow(heart_silhouette, cmap='gray'); ax_prog[1].set_title("5. Largest Blob and Smoothed"); ax_prog[1].grid(False)
        
        overlay_silhouette = np.stack((img_crop,)*3, axis=-1)
        overlay_silhouette[silhouette_edge] = [255, 0, 0] 
        ax_prog[2].imshow(overlay_silhouette); ax_prog[2].set_title("6. Closed-Loop Edge Overlay"); ax_prog[2].grid(False)
        fig_prog.tight_layout()

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
        ax_final[0].set_title("Organic Cardiac Edge on Full CXR")
        ax_final[0].grid(False)
        
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
        ax_final[1].grid(False)
        
        ax_final[2].imshow(extracted_heart, cmap='gray')
        ax_final[2].set_title("Extracted Heart")
        ax_final[2].grid(False)
        
        fig_final.tight_layout()

        # RETURN figures and dataframe for GUI insertion
        return df_features, fig_otsu, fig_prog, fig_final