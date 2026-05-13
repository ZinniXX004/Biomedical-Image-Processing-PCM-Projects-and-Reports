'''
Assignment 1 
- Segmentation for the Fetus Head of Ultrasound Image (PNG file format)
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
from skimage.filters import threshold_multiotsu
from skimage.measure import shannon_entropy, EllipseModel
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity
from skimage.draw import ellipse, ellipse_perimeter
import math

# Use white style for a clean background, removing default grid lines
sns.set_theme(style="white")

class UltrasoundProcessor:
    def __init__(self, path_img1, path_img2, crop_coords, export_dir):
        """Initialize ultrasound image dataset, crop coordinates, and export directory."""
        def load_gray(path):
            img = imageio.imread(path)
            if img.ndim == 3:
                return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            return img

        self.dataset = {
            "Fetus 166_HC": load_gray(path_img1),
            "Fetus 87_HC": load_gray(path_img2)
        }
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
        # Plot Histogram + Ogive (CDF) filtering out background pixels <= 15
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
        Display Original Grayscale Ultrasound Image and Histogram.
        Returns the Matplotlib Figure object.
        """
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
        
        axes1[0].imshow(img_gray, cmap='gray')
        axes1[0].set_title("Original Grayscale USG (Check coordinates here)")
        axes1[0].grid(False)
        
        self.plot_hist_with_ogive(img_gray.ravel(), axes1[1], 'gray', "Original Image Histogram")
        fig1.tight_layout()
        
        return fig1

    def step2_crop_and_restore(self, img_gray, title):
        # Crop and apply Median Filter for Speckle Noise reduction
        x_min, x_max, y_min, y_max = self.crop_coords[title]
        x_min, x_max = max(0, x_min), min(img_gray.shape[1], x_max)
        y_min, y_max = max(0, y_min), min(img_gray.shape[0], y_max)
        
        img_crop = img_gray[y_min:y_max, x_min:x_max]
        img_restored = ndi.median_filter(img_crop, size=5)
        
        print(f"-> Cropped at X:({x_min}-{x_max}), Y:({y_min}-{y_max}) and Despeckled.")
        return img_crop, img_restored, (x_min, x_max, y_min, y_max)

    def step3_evaluate_enhancement(self, img_restored, title):
        """
        GUI ADAPTATION:
        Evaluate performance, Bar Charts.
        REMOVED terminal input(). Returns a dictionary of images so GUI can choose later.
        """
        p2, p98 = np.percentile(img_restored, (2, 98))
        img_cs = exposure.rescale_intensity(img_restored, in_range=(p2, p98), out_range=(0, 255)).astype(np.uint8)
        img_he = (exposure.equalize_hist(img_restored) * 255).astype(np.uint8)
        img_clahe = (exposure.equalize_adapthist(img_restored, clip_limit=0.03) * 255).astype(np.uint8)

        methods = [("Contrast Stretching", img_cs), ("Histogram Eq", img_he), ("CLAHE", img_clahe)]
        metrics_data = []
        for name, img in methods:
            rmse, psnr, ssim, entropy = self.calc_metrics(img_restored, img)
            metrics_data.append({
                "Enhancement Method": name, 
                "RMSE (levels)": rmse, 
                "PSNR (dB)": psnr, 
                "SSIM": ssim, 
                "Entropy (bits)": entropy
            })
            
        df_metrics = pd.DataFrame(metrics_data)
        
        fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
        images = [img_restored, img_cs, img_he, img_clahe]
        titles = ["Restored (Despeckled)", "Contrast Stretching", "Histogram Eq", "CLAHE"]
        
        for i in range(4):
            axes2[0, i].imshow(images[i], cmap='gray'); axes2[0, i].set_title(titles[i], fontsize=12); axes2[0, i].grid(False) 
            self.plot_hist_with_ogive(images[i].ravel(), axes2[1, i], 'teal', f"Hist+Ogive: {titles[i]}")
        fig2.tight_layout()

        fig_bar, ax_bar = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.barplot(data=df_metrics, x="Enhancement Method", y="RMSE (levels)", hue="Enhancement Method", legend=False, ax=ax_bar[0,0], palette="Reds")
        ax_bar[0,0].set_title("RMSE (Lower is Better)"); ax_bar[0,0].set_ylabel("RMSE Score")
        sns.barplot(data=df_metrics, x="Enhancement Method", y="PSNR (dB)", hue="Enhancement Method", legend=False, ax=ax_bar[0,1], palette="Greens")
        ax_bar[0,1].set_title("PSNR (Higher is Better)"); ax_bar[0,1].set_ylabel("PSNR (dB)")
        sns.barplot(data=df_metrics, x="Enhancement Method", y="SSIM", hue="Enhancement Method", legend=False, ax=ax_bar[1,0], palette="Blues")
        ax_bar[1,0].set_title("SSIM (Closer to 1.0 is Better)"); ax_bar[1,0].set_ylabel("SSIM Score")
        sns.barplot(data=df_metrics, x="Enhancement Method", y="Entropy (bits)", hue="Enhancement Method", legend=False, ax=ax_bar[1,1], palette="Purples")
        ax_bar[1,1].set_title("Entropy (Tissue Texture Detail)"); ax_bar[1,1].set_ylabel("Shannon Entropy (bits/px)")
        fig_bar.tight_layout()

        # Dictionary to store the enhanced images so the GUI can pick one based on Combobox selection
        enhanced_images_dict = {
            "Contrast Stretching": img_cs,
            "Histogram Equalization": img_he,
            "CLAHE": img_clahe
        }

        return enhanced_images_dict, df_metrics, fig2, fig_bar

    def extract_fetal_features(self, a, b):
        """Calculate enhanced biometry features using mathematical ellipse parameters."""
        major = max(a, b) * 2
        minor = min(a, b) * 2
        ofd = major
        bpd = minor
        
        h = ((a - b)**2) / ((a + b)**2)
        hc = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
        ha = math.pi * a * b 
        ci = bpd / ofd if ofd > 0 else 0 
        
        eccentricity = math.sqrt(1 - (minor/major)**2) if major > 0 else 0
        eq_diameter = math.sqrt((4 * ha) / math.pi) if ha > 0 else 0
        circularity = (4 * math.pi * ha) / (hc**2) if hc > 0 else 0
        
        return {
            "BPD (px)": round(bpd, 2), "OFD (px)": round(ofd, 2), "HC (px)": round(hc, 2),
            "Area (px^2)": round(ha, 2), "Cephalic Index": round(ci, 4), "Eccentricity": round(eccentricity, 4),
            "Equivalent Diameter (px)": round(eq_diameter, 2), "Circularity": round(circularity, 4)
        }

    def step4_segment_and_extract(self, best_img, img_crop, img_full, coords, title, method_name):
        """
        GUI ADAPTATION:
        Multi-Otsu Segmentation, Ellipse Fitting, Overlays, and Features.
        Returns Pandas DataFrame and Figure objects.
        """
        x_min, x_max, y_min, y_max = coords
        
        # 1. MULTI-OTSU THRESHOLDING (3 Classes: BG, Brain, Skull)
        thresholds = threshold_multiotsu(best_img, classes=3)
        regions = np.digitize(best_img, bins=thresholds)
        
        class_0 = (regions == 0) # Background / Amniotic Fluid
        class_1 = (regions == 1) # Soft Tissue / Brain
        class_2 = (regions == 2) # Skull Bone
        
        skull_mask = morphology.remove_small_objects(class_2, min_size=50)
        skull_mask = morphology.binary_closing(skull_mask, morphology.disk(3))

        # 2. ELLIPSE FITTING MODEL
        ellipse_mask = np.zeros_like(img_crop, dtype=bool)
        ellipse_edge = np.zeros_like(img_crop, dtype=bool)
        
        points_y, points_x = np.nonzero(skull_mask)
        points_xy = np.column_stack((points_x, points_y))
        
        feat_dict = {
            "BPD (px)": 0, "OFD (px)": 0, "HC (px)": 0, "Area (px^2)": 0, 
            "Cephalic Index": 0, "Eccentricity": 0, "Equivalent Diameter (px)": 0, 
            "Circularity": 0, "Mean Brain Echogenicity (0-255)": 0, "Brain Tissue Variance": 0
        }
        
        xc, yc, a, b, theta = 0, 0, 0, 0, 0
        
        if len(points_xy) > 10:
            ell = EllipseModel()
            success = ell.estimate(points_xy)
            if success:
                xc, yc, a, b, theta = ell.params
                
                rr, cc = ellipse(yc, xc, b, a, shape=img_crop.shape, rotation=-theta)
                ellipse_mask[rr, cc] = True
                
                rr_p, cc_p = ellipse_perimeter(int(yc), int(xc), int(b), int(a), orientation=theta)
                valid = (rr_p >= 0) & (rr_p < img_crop.shape[0]) & (cc_p >= 0) & (cc_p < img_crop.shape[1])
                ellipse_edge[rr_p[valid], cc_p[valid]] = True
                
                brain_pixels = img_crop[ellipse_mask]
                mean_brain = np.mean(brain_pixels) if len(brain_pixels) > 0 else 0
                var_brain = np.var(brain_pixels) if len(brain_pixels) > 0 else 0
                
                geom_features = self.extract_fetal_features(a, b)
                feat_dict.update(geom_features)
                feat_dict["Mean Brain Echogenicity (0-255)"] = round(mean_brain, 2)
                feat_dict["Brain Tissue Variance"] = round(var_brain, 2)

        # 3. FEATURE EXTRACTION TABLE EXPORT
        descriptions = {
            "BPD (px)": "Biparietal Diameter (Minor Axis). Key for Gestational Age.",
            "OFD (px)": "Occipitofrontal Diameter (Major Axis).",
            "HC (px)": "Head Circumference (Ramanujan Approximation).",
            "Area (px^2)": "Fetal Head Area within the skull.",
            "Cephalic Index": "Ratio of BPD/OFD. Normal range: 0.75 - 0.85.",
            "Eccentricity": "Deviation from circularity (0 = circle, approaching 1 = elongated line).",
            "Equivalent Diameter (px)": "Diameter of a circle with the same area as the head.",
            "Circularity": "Perfect circle ratio (4*pi*Area / HC^2).",
            "Mean Brain Echogenicity (0-255)": "Average pixel brightness inside the skull (Fluid vs Tissue).",
            "Brain Tissue Variance": "Pixel spread/homogeneity inside the skull."
        }

        features_data = [{"Biometric Feature": key, "Description": descriptions[key], "Measured Value": val} for key, val in feat_dict.items()]
        df_features = pd.DataFrame(features_data)
        
        export_file = os.path.join(self.export_dir, f"USG_Features_{title}.xlsx")
        df_features.to_excel(export_file, index=False)
        print(f"[SUCCESS] Feature data successfully exported to: {export_file}")

        # 4. PLOTTING VISUALIZATIONS
        fig_otsu, ax_otsu = plt.subplots(2, 2, figsize=(12, 10))
        ax_otsu[0, 0].imshow(best_img, cmap='gray'); ax_otsu[0, 0].set_title("Selected Enhanced Image"); ax_otsu[0, 0].grid(False)
        ax_otsu[0, 1].imshow(class_0, cmap='gray'); ax_otsu[0, 1].set_title("Class 0: Amniotic Fluid (Background)"); ax_otsu[0, 1].grid(False)
        ax_otsu[1, 0].imshow(class_1, cmap='gray'); ax_otsu[1, 0].set_title("Class 1: Brain / Soft Tissue"); ax_otsu[1, 0].grid(False)
        ax_otsu[1, 1].imshow(class_2, cmap='gray'); ax_otsu[1, 1].set_title("Class 2: Skull Bone (Highlight)"); ax_otsu[1, 1].grid(False)
        fig_otsu.tight_layout()

        alpha_mask = np.zeros_like(img_full, dtype=float)
        alpha_mask[y_min:y_max, x_min:x_max] = 0.5
        full_regions = np.zeros_like(img_full, dtype=np.uint8)
        full_regions[y_min:y_max, x_min:x_max] = regions

        fig_class_over, ax_class_over = plt.subplots(1, 2, figsize=(16, 7))
        ax_class_over[0].imshow(img_crop, cmap='gray')
        ax_class_over[0].imshow(regions, cmap='viridis', alpha=0.5)
        ax_class_over[0].set_title("Multi-Otsu Overlay (Cropped)")
        ax_class_over[0].grid(False)
        
        ax_class_over[1].imshow(img_full, cmap='gray')
        ax_class_over[1].imshow(full_regions, cmap='viridis', alpha=alpha_mask)
        ax_class_over[1].set_title("Multi-Otsu Overlay (Full Size)")
        ax_class_over[1].grid(False)
        fig_class_over.tight_layout()

        # 5. Final Segmentation (Filled Mask and Extraction)
        filled_white_mask = np.zeros_like(img_crop, dtype=np.uint8)
        filled_white_mask[ellipse_mask] = 255 # Pure White inside ROI
        
        extracted_head = np.zeros_like(img_crop)
        extracted_head[ellipse_mask] = img_crop[ellipse_mask] 

        overlay_crop_edge = np.stack((img_crop,)*3, axis=-1)
        overlay_crop_edge[ellipse_edge] = [0, 255, 0] # Green Ellipse Edge
        
        fig_final, ax_final = plt.subplots(1, 3, figsize=(18, 6))
        
        ax_final[0].imshow(overlay_crop_edge)
        ax_final[0].set_title("Ellipse Edge Overlay")
        ax_final[0].grid(False)
        
        ax_final[1].imshow(filled_white_mask, cmap='gray')
        ax_final[1].set_title("Solid Filled ROI Mask (White Inner Area)")
        ax_final[1].grid(False)
        
        ax_final[2].imshow(extracted_head, cmap='gray')
        ax_final[2].set_title("Extracted Fetus Head")
        ax_final[2].grid(False)
        
        if feat_dict["Area (px^2)"] > 0:
            ax_final[0].plot(xc, yc, marker='o', color='yellow', markersize=6, label='Centroid')
            
            # Project both radii correctly onto the 2D image plane using angle theta
            vec_x = (math.cos(theta) * a, math.sin(theta) * a)
            vec_y = (-math.sin(theta) * b, math.cos(theta) * b)
            
            # Automatically assign the longest vector to Major (OFD) and shortest to Minor (BPD)
            if a >= b:
                vec_major = vec_x
                vec_minor = vec_y
            else:
                vec_major = vec_y
                vec_minor = vec_x
                
            x_major, y_major = xc + vec_major[0], yc + vec_major[1]
            x_minor, y_minor = xc + vec_minor[0], yc + vec_minor[1]
            
            ax_final[0].plot((xc, x_minor), (yc, y_minor), '-r', linewidth=2, label='BPD (Minor)')
            ax_final[0].plot((xc, x_major), (yc, y_major), '-m', linewidth=2, label='OFD (Major)')
            ax_final[0].legend(loc='upper right', fontsize=8)

        fig_final.tight_layout()

        # RETURN figures and dataframe for GUI insertion
        return df_features, fig_otsu, fig_class_over, fig_final

# Note: The CLI testing block (__main__) is removed since this is now a GUI library component