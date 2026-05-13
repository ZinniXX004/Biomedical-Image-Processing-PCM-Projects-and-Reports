'''
Final Project 1 
- Segmentation for the Optic Disc of Fundus Retina Image (JPG file format)
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
from skimage import exposure, measure, morphology
from skimage.filters import threshold_otsu
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity
from skimage.draw import disk
import math

# Use white style for a clean background, removing default grid lines
sns.set_theme(style="white")

class FundusProcessor:
    def __init__(self, path_normal, path_retino, crop_coords, export_dir):
        # Initialize image dataset, crop coordinates, and export directory
        self.dataset = {
            "Normal Fundus": imageio.imread(path_normal),
            "Retinography": imageio.imread(path_retino)
        }
        self.crop_coords = crop_coords
        self.export_dir = export_dir

        # Create export directory if it doesn't exist
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def calc_metrics(self, img_orig, img_enh):
        # Calculate RMSE, PSNR, SSIM, and Shannon Entropy metrics
        orig_norm = img_orig.astype(np.float64)
        enh_norm = img_enh.astype(np.float64)
        
        rmse = np.sqrt(mean_squared_error(orig_norm, enh_norm))
        psnr = peak_signal_noise_ratio(orig_norm, enh_norm, data_range=255)
        ssim = structural_similarity(img_orig, img_enh, data_range=255)
        entropy = shannon_entropy(img_enh)
        
        return round(rmse, 2), round(psnr, 2), round(ssim, 4), round(entropy, 4)

    def plot_hist_with_ogive(self, data, ax, color, title):
        # Helper Function: Plot Histogram + Ogive (CDF) via NumPy (Ignoring Background <= 15)
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

    def step1_plot_and_convert_channels(self, img_rgb, title):
        """
        GUI ADAPTATION:
        Converts image to Red Channel and generates figures.
        Returns the red channel image and the Matplotlib Figure objects.
        """
        img_red = img_rgb[:, :, 0]
        
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
        axes1[0].imshow(img_rgb); axes1[0].set_title("Original RGB (Check pixel coordinates here)"); axes1[0].grid(False) 
        axes1[1].imshow(img_red, cmap='gray'); axes1[1].set_title("Red Channel Conversion"); axes1[1].grid(False)
        fig1.tight_layout()
        
        fig_hist, axes_hist = plt.subplots(1, 4, figsize=(20, 5))
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            self.plot_hist_with_ogive(img_rgb[:, :, i].ravel(), axes_hist[i], color, f"Original {color.capitalize()} Channel")
        self.plot_hist_with_ogive(img_red.ravel(), axes_hist[3], 'gray', "Converted Red Channel")
        fig_hist.tight_layout()

        # RETURN figures instead of plt.show()
        return img_red, fig1, fig_hist

    def step2_hardcoded_crop(self, img_red, img_rgb, title):
        # Crop using predefined X and Y coordinates
        x_min, x_max, y_min, y_max = self.crop_coords[title]
        x_min, x_max = max(0, x_min), min(img_red.shape[1], x_max)
        y_min, y_max = max(0, y_min), min(img_red.shape[0], y_max)
        
        crop_red = img_red[y_min:y_max, x_min:x_max]
        crop_rgb = img_rgb[y_min:y_max, x_min:x_max]
        
        return crop_red, crop_rgb, (x_min, x_max, y_min, y_max)

    def step3_evaluate_enhancement(self, crop_red, title):
        """
        GUI ADAPTATION:
        Evaluates enhancement methods, creates metric dataframes and figures.
        REMOVED terminal input(). Returns a dictionary of images so GUI can choose later.
        """
        p2, p98 = np.percentile(crop_red, (2, 98))
        img_cs = exposure.rescale_intensity(crop_red, in_range=(p2, p98), out_range=(0, 255)).astype(np.uint8)
        img_he = (exposure.equalize_hist(crop_red) * 255).astype(np.uint8)
        img_clahe = (exposure.equalize_adapthist(crop_red, clip_limit=0.03) * 255).astype(np.uint8)

        methods = [("Contrast Stretching", img_cs), ("Histogram Eq", img_he), ("CLAHE", img_clahe)]
        metrics_data = []
        for name, img in methods:
            rmse, psnr, ssim, entropy = self.calc_metrics(crop_red, img)
            metrics_data.append({
                "Enhancement Method": name, 
                "RMSE (levels)": rmse, 
                "PSNR (dB)": psnr, 
                "SSIM": ssim, 
                "Entropy (bits)": entropy
            })
            
        df_metrics = pd.DataFrame(metrics_data)
        
        fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
        images = [crop_red, img_cs, img_he, img_clahe]
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
        ax_bar[1,0].set_title("SSIM (Closer to 1.0 is Better)")
        
        sns.barplot(data=df_metrics, x="Enhancement Method", y="Entropy (bits)", hue="Enhancement Method", legend=False, ax=ax_bar[1,1], palette="Purples")
        ax_bar[1,1].set_title("Entropy (Vascular Detail and Texture)")
        fig_bar.tight_layout()

        # Dictionary to store the enhanced images so the GUI can pick one based on Combobox selection
        enhanced_images_dict = {
            "Contrast Stretching": img_cs,
            "Histogram Equalization": img_he,
            "CLAHE": img_clahe
        }

        # Return figures and the dictionary, NO input() prompt
        return enhanced_images_dict, df_metrics, fig2, fig_bar

    def get_features(self, od_mask, oc_mask, crop_red):
        """Helper function to extract 12 clinical and statistical features."""
        labeled_od = measure.label(od_mask)
        props_od = measure.regionprops(labeled_od, intensity_image=crop_red)[0] if labeled_od.max() > 0 else None
        labeled_oc = measure.label(oc_mask)
        props_oc = measure.regionprops(labeled_oc)[0] if labeled_oc.max() > 0 else None

        if not props_od:
            return {k: 0 for k in ['OD Area (px^2)', 'Cup Area (px^2)', 'NRR Area (px^2)', 'CDR (Ratio)', 'Circularity', 'Mean Int (0-255)', 'Variance', 'Skewness', 'Kurtosis', 'Entropy (bits)', 'GLCM Contrast', 'GLCM Homogeneity']}

        area_od = props_od.area
        area_oc = props_oc.area if props_oc else 0
        nrr_area = area_od - area_oc
        cdr = np.sqrt(area_oc / area_od) if area_od > 0 else 0
        perimeter = props_od.perimeter
        circularity = (4 * math.pi * area_od) / (perimeter ** 2) if perimeter > 0 else 0
        
        od_pixels = crop_red[od_mask]
        mean_int = props_od.mean_intensity
        var_int = np.var(od_pixels) if len(od_pixels) > 0 else 0
        skew_val = skew(od_pixels) if len(od_pixels) > 0 else 0
        kurt_val = kurtosis(od_pixels) if len(od_pixels) > 0 else 0
        
        entropy_val = shannon_entropy(od_pixels) if len(od_pixels) > 0 else 0
        
        min_row, min_col, max_row, max_col = props_od.bbox
        od_bbox_img = crop_red[min_row:max_row, min_col:max_col]
        
        glcm_contrast = 0; glcm_homog = 0
        if od_bbox_img.size > 0:
            glcm = graycomatrix(od_bbox_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
            glcm_homog = graycoprops(glcm, 'homogeneity')[0, 0]

        return {
            "OD Area (px^2)": round(area_od, 2),
            "Cup Area (px^2)": round(area_oc, 2),
            "NRR Area (px^2)": round(nrr_area, 2),
            "CDR (Ratio)": round(cdr, 4),
            "Circularity": round(circularity, 4),
            "Mean Int (0-255)": round(mean_int, 2),
            "Variance": round(var_int, 2),
            "Skewness": round(skew_val, 4),
            "Kurtosis": round(kurt_val, 4),
            "Entropy (bits)": round(entropy_val, 4),
            "GLCM Contrast": round(glcm_contrast, 4),
            "GLCM Homogeneity": round(glcm_homog, 4)
        }

    def step4_segment_and_extract(self, best_img, crop_red, crop_rgb, img_rgb_full, coords, title, method_name):
        """
        GUI ADAPTATION:
        Performs segmentation, creates plots, and saves to Excel.
        Returns Pandas DataFrame and Figure objects.
        """
        x_min, x_max, y_min, y_max = coords
        
        selem = morphology.disk(15)
        img_smoothed = morphology.closing(best_img, selem)
        
        thresh = threshold_otsu(img_smoothed)
        od_mask_initial = img_smoothed > thresh
        
        labeled_od = measure.label(od_mask_initial)
        largest_od = max(measure.regionprops(labeled_od), key=lambda x: x.area)
        
        # METHOD 1: NATURAL SHAPE
        od_mask_nat = morphology.convex_hull_image(labeled_od == largest_od.label)
        cup_thresh_nat = np.percentile(crop_red[od_mask_nat], 85) 
        oc_mask_nat = ndi.binary_fill_holes((crop_red > cup_thresh_nat) & od_mask_nat)
        labeled_oc_nat = measure.label(oc_mask_nat)
        if labeled_oc_nat.max() > 0:
            largest_oc_nat = max(measure.regionprops(labeled_oc_nat), key=lambda x: x.area)
            oc_mask_nat = morphology.convex_hull_image(labeled_oc_nat == largest_oc_nat.label)

        # METHOD 2: PERFECT CIRCLE
        od_mask_circ = np.zeros_like(crop_red, dtype=bool)
        oc_mask_circ = np.zeros_like(crop_red, dtype=bool)
        
        r0, c0 = largest_od.centroid
        radius_od = largest_od.equivalent_diameter / 2.0
        rr_od, cc_od = disk((r0, c0), radius_od, shape=crop_red.shape)
        od_mask_circ[rr_od, cc_od] = True
        
        if labeled_oc_nat.max() > 0:
            r0_cup, c0_cup = largest_oc_nat.centroid
            radius_oc = largest_oc_nat.equivalent_diameter / 2.0
            rr_oc, cc_oc = disk((r0_cup, c0_cup), radius_oc, shape=crop_red.shape)
            oc_mask_circ[rr_oc, cc_oc] = True

        # FEATURE EXTRACTION and COMPARISON TABLE
        feat_nat = self.get_features(od_mask_nat, oc_mask_nat, crop_red)
        feat_circ = self.get_features(od_mask_circ, oc_mask_circ, crop_red)

        descriptions = {
            "OD Area (px^2)": "Total area of Optic Disc.",
            "Cup Area (px^2)": "Total area of Optic Cup.",
            "NRR Area (px^2)": "Neuro-Retinal Rim Area (OD - Cup).",
            "CDR (Ratio)": "Cup-to-Disc Ratio.",
            "Circularity": "Anatomical roundness.",
            "Mean Int (0-255)": "Average OD pixel brightness.",
            "Variance": "Brightness intensity spread.",
            "Skewness": "Asymmetry of intensity.",
            "Kurtosis": "Peakedness of intensity.",
            "Entropy (bits)": "Information complexity and texture.",
            "GLCM Contrast": "Local contrast and depth.",
            "GLCM Homogeneity": "Uniformity of local pixel textures."
        }

        features_data = []
        for key in feat_nat.keys():
            features_data.append({
                "Feature Name": key,
                "Brief Description": descriptions[key],
                "Natural (Convex Hull)": feat_nat[key],
                "Perfect Circle": feat_circ[key]
            })

        df_features = pd.DataFrame(features_data)
        
        # EXPORT TO EXCEL
        export_file = os.path.join(self.export_dir, f"Features_{title}.xlsx")
        df_features.to_excel(export_file, index=False)

        # GENERATE EDGES and PLOTS
        edge_od_nat = morphology.binary_dilation(od_mask_nat, morphology.disk(2)) ^ od_mask_nat
        edge_oc_nat = morphology.binary_dilation(oc_mask_nat, morphology.disk(2)) ^ oc_mask_nat
        edge_od_circ = morphology.binary_dilation(od_mask_circ, morphology.disk(2)) ^ od_mask_circ
        edge_oc_circ = morphology.binary_dilation(oc_mask_circ, morphology.disk(2)) ^ oc_mask_circ

        full_mask_nat = np.zeros(img_rgb_full.shape[:2], dtype=bool)
        full_mask_circ = np.zeros(img_rgb_full.shape[:2], dtype=bool)
        full_mask_nat[y_min:y_max, x_min:x_max] = od_mask_nat
        full_mask_circ[y_min:y_max, x_min:x_max] = od_mask_circ

        full_edge_od_nat = np.zeros(img_rgb_full.shape[:2], dtype=bool)
        full_edge_oc_nat = np.zeros(img_rgb_full.shape[:2], dtype=bool)
        full_edge_od_circ = np.zeros(img_rgb_full.shape[:2], dtype=bool)
        full_edge_oc_circ = np.zeros(img_rgb_full.shape[:2], dtype=bool)
        
        full_edge_od_nat[y_min:y_max, x_min:x_max] = edge_od_nat
        full_edge_oc_nat[y_min:y_max, x_min:x_max] = edge_oc_nat
        full_edge_od_circ[y_min:y_max, x_min:x_max] = edge_od_circ
        full_edge_oc_circ[y_min:y_max, x_min:x_max] = edge_oc_circ

        # Plot 3: Masking
        fig_mask, ax_mask = plt.subplots(2, 2, figsize=(14, 12))
        ax_mask[0, 0].imshow(od_mask_nat, cmap='gray'); ax_mask[0, 0].set_title("Natural Mask (Cropped)"); ax_mask[0, 0].grid(False)
        ax_mask[0, 1].imshow(full_mask_nat, cmap='gray'); ax_mask[0, 1].set_title("Natural Mask (Full Size)"); ax_mask[0, 1].grid(False)
        ax_mask[1, 0].imshow(od_mask_circ, cmap='gray'); ax_mask[1, 0].set_title("Circle Mask (Cropped)"); ax_mask[1, 0].grid(False)
        ax_mask[1, 1].imshow(full_mask_circ, cmap='gray'); ax_mask[1, 1].set_title("Circle Mask (Full Size)"); ax_mask[1, 1].grid(False)
        fig_mask.tight_layout()

        # Overlays
        overlay_crop_red_nat = np.stack((crop_red,)*3, axis=-1)
        overlay_crop_red_nat[edge_od_nat] = [0, 255, 0]; overlay_crop_red_nat[edge_oc_nat] = [255, 0, 0]
        
        overlay_crop_red_circ = np.stack((crop_red,)*3, axis=-1)
        overlay_crop_red_circ[edge_od_circ] = [0, 255, 0]; overlay_crop_red_circ[edge_oc_circ] = [255, 0, 0]

        overlay_crop_nat = crop_rgb.copy()
        overlay_crop_nat[edge_od_nat] = [0, 255, 0]; overlay_crop_nat[edge_oc_nat] = [255, 0, 0] 
        
        overlay_crop_circ = crop_rgb.copy()
        overlay_crop_circ[edge_od_circ] = [0, 255, 0]; overlay_crop_circ[edge_oc_circ] = [255, 0, 0]

        overlay_full_nat = img_rgb_full.copy()
        overlay_full_nat[full_edge_od_nat] = [0, 255, 0]; overlay_full_nat[full_edge_oc_nat] = [255, 0, 0]
        
        overlay_full_circ = img_rgb_full.copy()
        overlay_full_circ[full_edge_od_circ] = [0, 255, 0]; overlay_full_circ[full_edge_oc_circ] = [255, 0, 0]

        # Plot 4: Overlays
        fig_over, ax_over = plt.subplots(2, 3, figsize=(20, 12))
        
        ax_over[0, 0].imshow(overlay_crop_red_nat); ax_over[0, 0].set_title("Natural Overlay (Red Channel Cropped)"); ax_over[0, 0].grid(False)
        ax_over[0, 1].imshow(overlay_crop_nat); ax_over[0, 1].set_title("Natural Overlay (RGB Cropped)"); ax_over[0, 1].grid(False)
        
        labeled_od_nat = measure.label(od_mask_nat)
        if labeled_od_nat.max() > 0:
            props_od_nat = measure.regionprops(labeled_od_nat)[0]
            y0, x0 = props_od_nat.centroid
            ax_over[0, 1].plot(x0, y0, marker='o', color='yellow', markersize=8, label='Centroid')
            
            orientation = props_od_nat.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props_od_nat.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props_od_nat.minor_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props_od_nat.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props_od_nat.major_axis_length
            
            ax_over[0, 1].plot((x0, x1), (y0, y1), '-r', linewidth=2, label='Minor Axis')
            ax_over[0, 1].plot((x0, x2), (y0, y2), '-m', linewidth=2, label='Major Axis')
            
            minr, minc, maxr, maxc = props_od_nat.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax_over[0, 1].plot(bx, by, '-c', linewidth=2, label='Bounding Box')
            ax_over[0, 1].legend(loc='upper right', fontsize=8)

        ax_over[0, 2].imshow(overlay_full_nat); ax_over[0, 2].set_title("Natural Overlay (RGB Full Size)"); ax_over[0, 2].grid(False)

        ax_over[1, 0].imshow(overlay_crop_red_circ); ax_over[1, 0].set_title("Circle Overlay (Red Channel Cropped)"); ax_over[1, 0].grid(False)
        ax_over[1, 1].imshow(overlay_crop_circ); ax_over[1, 1].set_title("Circle Overlay (RGB Cropped)"); ax_over[1, 1].grid(False)
        ax_over[1, 2].imshow(overlay_full_circ); ax_over[1, 2].set_title("Circle Overlay (RGB Full Size)"); ax_over[1, 2].grid(False)
        
        fig_over.tight_layout()

        # RETURN figures and dataframe for GUI insertion
        return df_features, fig_mask, fig_over

# Note: The CLI testing block (__main__) is removed since this is now a GUI library component