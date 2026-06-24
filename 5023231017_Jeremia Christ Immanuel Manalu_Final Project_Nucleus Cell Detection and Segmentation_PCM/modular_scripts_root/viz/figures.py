"""Matplotlib figure factories for every visualization tab"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from ..ui.theme import PAL


def _ax_off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def create_histogram_figure(image_rgb: np.ndarray, H_u8: np.ndarray, title: str) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"RGB & H-Channel Histogram  ·  {title}", fontsize=11, fontweight="bold",
                 color=PAL["text0"])

    axes[0, 0].imshow(image_rgb); axes[0, 0].set_title("Original H&E", color=PAL["text0"])
    axes[0, 1].imshow(H_u8, cmap="gray"); axes[0, 1].set_title("H Channel (extracted)", color=PAL["text0"])
    _ax_off(axes[0, 0]); _ax_off(axes[0, 1])

    ax = axes[1, 0]
    for ch, col, lbl in zip([0, 1, 2], ["#F44336", "#66BB6A", "#42A5F5"], ["R", "G", "B"]):
        hist, bins = np.histogram(image_rgb[:, :, ch].ravel(), bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=col, alpha=0.85, linewidth=1.4, label=lbl)
    ax.set_title("RGB Channel Histograms", color=PAL["text0"])
    ax.set_xlabel("Pixel Value"); ax.set_ylabel("Count")
    ax.legend(fontsize=9); ax.set_xlim(0, 255)
    ax.tick_params(colors=PAL["text1"]); ax.set_facecolor(PAL["bg3"])

    ax2 = axes[1, 1]
    hist_h, bins_h = np.histogram(H_u8.ravel(), bins=256, range=(0, 256))
    ax2.bar(bins_h[:-1], hist_h, width=1, color="#9575CD", alpha=0.85, label="H-channel")
    otsu_val, _ = cv2.threshold(H_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pct75       = np.percentile(H_u8, 75)
    ax2.axvline(otsu_val, color="#F44336", lw=2, linestyle="--",
                label=f"Otsu = {int(otsu_val)}")
    ax2.axvline(pct75, color="#FF9800", lw=2, linestyle="-.",
                label=f"75th-pct = {pct75:.0f}")
    ax2.set_title("H-Channel Histogram + Threshold Candidates", color=PAL["text0"])
    ax2.set_xlabel("Pixel Value"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=9); ax2.set_xlim(0, 255)
    ax2.tick_params(colors=PAL["text1"]); ax2.set_facecolor(PAL["bg3"])

    fig.tight_layout()
    return fig


def create_clahe_figure(H_u8_raw: np.ndarray, params: dict, title: str) -> Figure:
    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip_limit"],
                              tileGridSize=params["clahe_tile_size"])
    H_cl  = clahe.apply(H_u8_raw)
    diff  = H_cl.astype(np.int16) - H_u8_raw.astype(np.int16)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"CLAHE Diagnostic  ·  {title}\n"
        f"clipLimit={params['clahe_clip_limit']}  tileGridSize={params['clahe_tile_size']}",
        fontsize=11, fontweight="bold", color=PAL["text0"])

    axes[0, 0].imshow(H_u8_raw, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Before CLAHE", color=PAL["text0"]); _ax_off(axes[0, 0])
    axes[0, 1].imshow(H_cl, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("After CLAHE", color=PAL["text0"]); _ax_off(axes[0, 1])
    im = axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-80, vmax=80)
    axes[0, 2].set_title("Difference (After − Before)", color=PAL["text0"]); _ax_off(axes[0, 2])
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    bins  = np.arange(257)
    h_bef = np.histogram(H_u8_raw.ravel(), bins=bins)[0].astype(float)
    h_aft = np.histogram(H_cl.ravel(),     bins=bins)[0].astype(float)

    for ax, data, color, title_lbl in [
        (axes[1, 0], h_bef, "#42A5F5", "Histogram — Before CLAHE"),
        (axes[1, 1], h_aft, "#FF9800",  "Histogram — After CLAHE"),
    ]:
        ax.bar(bins[:-1], data, width=1, color=color, alpha=0.85)
        ax.set_title(title_lbl, color=PAL["text0"])
        ax.set_xlabel("Pixel Value"); ax.set_ylabel("Count"); ax.set_xlim(0, 255)
        ax.tick_params(colors=PAL["text1"]); ax.set_facecolor(PAL["bg3"])

    cdf_bef = np.cumsum(h_bef) / h_bef.sum()
    cdf_aft = np.cumsum(h_aft) / h_aft.sum()
    x       = bins[:-1]
    ax3 = axes[1, 2]
    ax3.plot(x, cdf_bef, color="#42A5F5", lw=2, label="Before CLAHE")
    ax3.plot(x, cdf_aft, color="#FF9800", lw=2, label="After CLAHE")
    ax3.plot([0, 255], [0, 1], color=PAL["text2"], lw=1, linestyle="--", alpha=0.6,
             label="Ideal (uniform)")
    ax3.set_title("CDF Comparison\n(closer to diagonal = more uniform)", color=PAL["text0"])
    ax3.set_xlabel("Pixel Value"); ax3.set_ylabel("CDF")
    ax3.legend(fontsize=9); ax3.set_xlim(0, 255)
    ax3.tick_params(colors=PAL["text1"]); ax3.set_facecolor(PAL["bg3"])

    dstd = H_cl.std() - H_u8_raw.std()
    fig.text(0.5, 0.01,
             f"Std: {H_u8_raw.std():.1f} → {H_cl.std():.1f}  (Δ={dstd:+.1f})   "
             f"Dynamic range: {H_cl.min()}–{H_cl.max()}",
             ha="center", fontsize=10, color=PAL["text2"])
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def create_segmentation_figure(image_rgb, gt_mask, pred_mask, metrics, title) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"{title}\n"
        f"IoU={metrics['IoU']:.4f}  Dice={metrics['Dice']:.4f}  "
        f"Prec={metrics['Precision']:.4f}  Rec={metrics['Recall']:.4f}",
        fontsize=11, fontweight="bold", color=PAL["text0"])

    def blend(img, mask, colour, alpha=0.55):
        ov = img.astype(np.float32).copy()
        ov[mask > 0] = ov[mask > 0] * (1 - alpha) + np.array(colour) * alpha
        return np.clip(ov, 0, 255).astype(np.uint8)

    axes[0, 0].imshow(image_rgb);  axes[0, 0].set_title("Original H&E",     color=PAL["text0"])
    axes[0, 1].imshow(gt_mask,   cmap="Greens");  axes[0, 1].set_title("Ground Truth",     color=PAL["text0"])
    axes[0, 2].imshow(pred_mask, cmap="Oranges"); axes[0, 2].set_title("Predicted Mask",   color=PAL["text0"])
    axes[1, 0].imshow(blend(image_rgb, gt_mask,   [0, 220, 0]));   axes[1, 0].set_title("GT Overlay",         color=PAL["text0"])
    axes[1, 1].imshow(blend(image_rgb, pred_mask, [255, 140, 0])); axes[1, 1].set_title("Prediction Overlay", color=PAL["text0"])

    g  = gt_mask > 0; p = pred_mask > 0
    em = np.zeros((*gt_mask.shape, 3), np.uint8)
    em[np.logical_and( p,  g)] = [0,  210,   0]
    em[np.logical_and( p, ~g)] = [220,  0,   0]
    em[np.logical_and(~p,  g)] = [0,    0, 220]
    axes[1, 2].imshow(em)
    axes[1, 2].set_title("Error Map  (TP / FP / FN)", color=PAL["text0"])
    axes[1, 2].legend(
        handles=[mpatches.Patch(color="#00D200", label=f"TP {metrics['TP']:,}"),
                 mpatches.Patch(color="#DC0000", label=f"FP {metrics['FP']:,}"),
                 mpatches.Patch(color="#0000DC", label=f"FN {metrics['FN']:,}")],
        loc="lower right", fontsize=9)

    for ax in axes.flat:
        _ax_off(ax)
    fig.tight_layout()
    return fig


def create_stepwise_bar_figure(stages, ious, dices, title) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Step-by-Step Performance  ·  {title}", fontsize=11, fontweight="bold",
                 color=PAL["text0"])

    x     = np.arange(len(stages))
    width = 0.38
    ci    = ["#5C9BD4"] * (len(x) - 1) + ["#1565C0"]
    cd    = ["#FFA040"] * (len(x) - 1) + ["#E65100"]

    ax = axes[0]
    bi = ax.bar(x - width / 2, ious,  width, color=ci, edgecolor="#1A3A60", lw=0.7, label="IoU")
    bd = ax.bar(x + width / 2, dices, width, color=cd, edgecolor="#1A3A60", lw=0.7, label="Dice")
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=8, color=PAL["text1"])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Absolute Score per Stage", color=PAL["text0"])
    ax.legend(fontsize=9); ax.tick_params(colors=PAL["text1"]); ax.set_facecolor(PAL["bg3"])
    for bar, val in zip(list(bi) + list(bd), ious + dices):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=7.5, fontweight="bold", color=PAL["text0"])

    ax2 = axes[1]
    di  = [v - ious[0]  for v in ious]
    dd  = [v - dices[0] for v in dices]
    ci2 = ["#43A047" if d >= 0 else "#E53935" for d in di]
    cd2 = ["#FB8C00" if d >= 0 else "#8E24AA" for d in dd]
    ax2.bar(x - width / 2, di, width, color=ci2, edgecolor="#1A3A60", lw=0.7, label="ΔIoU")
    ax2.bar(x + width / 2, dd, width, color=cd2, edgecolor="#1A3A60", lw=0.7, label="ΔDice")
    ax2.axhline(0, color=PAL["text2"], lw=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(stages, fontsize=8, color=PAL["text1"])
    ax2.set_ylabel("Δ vs Stage ①")
    ax2.set_title("Incremental Contribution", color=PAL["text0"])
    ax2.legend(fontsize=9); ax2.tick_params(colors=PAL["text1"]); ax2.set_facecolor(PAL["bg3"])

    fig.tight_layout()
    return fig


def create_stepwise_grid_figure(image_rgb, gt_mask, masks, stages, ious, dices, title) -> Figure:
    n_cols = len(stages)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 3.0, 9))
    fig.suptitle(f"Intermediate Mask Grid  ·  {title}", fontsize=11, fontweight="bold",
                 color=PAL["text0"])
    gt_b = gt_mask > 0

    for j, (sname, mask_s, iou_s, dice_s) in enumerate(zip(stages, masks, ious, dices)):
        axes[0, j].imshow(mask_s, cmap="gray")
        axes[0, j].set_title(sname, fontsize=7.5, color=PAL["text0"])

        pb  = mask_s > 0
        err = np.zeros((*mask_s.shape, 3), np.uint8)
        err[np.logical_and( pb,  gt_b)] = [0,   200,   0]
        err[np.logical_and( pb, ~gt_b)] = [220,   0,   0]
        err[np.logical_and(~pb,  gt_b)] = [0,     0, 220]
        axes[1, j].imshow(err)
        axes[1, j].set_title(f"IoU={iou_s:.3f}\nDice={dice_s:.3f}", fontsize=7.5, color=PAL["text0"])

        ov = image_rgb.copy().astype(np.float32)
        ov[mask_s > 0] = ov[mask_s > 0] * 0.45 + np.array([255, 165, 0]) * 0.55
        axes[2, j].imshow(ov.astype(np.uint8))
        axes[2, j].set_title("Overlay", fontsize=7, color=PAL["text0"])

    for ax in axes.flat:
        _ax_off(ax)
    fig.tight_layout()
    return fig


def create_timing_figure(stages, times_ms, total_ms, n_cc, title) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    clrs = ["#42A5F5"] * len(stages)
    clrs[-1] = "#FF6F00"
    bars = ax.barh(stages, times_ms, color=clrs, edgecolor=PAL["border"], lw=0.7)
    ax.set_xlabel("Time (ms)", color=PAL["text1"])
    ax.set_title(
        f"Per-Stage Timing  ·  {title}\n"
        f"Total ≈ {total_ms:.1f} ms   |   CC before size-filter: {n_cc}",
        fontsize=11, color=PAL["text0"])
    ax.set_facecolor(PAL["bg3"]); ax.tick_params(colors=PAL["text1"]); ax.invert_yaxis()
    if times_ms:
        mx = max(times_ms)
        for bar, v in zip(bars, times_ms):
            ax.text(bar.get_width() + mx * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f} ms ({v / total_ms * 100:.0f}%)" if total_ms > 0 else f"{v:.1f}",
                    va="center", fontsize=8.5, color=PAL["text0"])
    fig.tight_layout()
    return fig


def create_summary_figure(df: pd.DataFrame) -> Figure:
    metrics = ["IoU", "Dice", "Precision", "Recall"]
    short   = [name.split("-")[1] + "\n" + name.split("-")[2][:5] for name in df["Image"]]
    colours = ["#42A5F5", "#66BB6A", "#FF9800", "#E91E63"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Segmentation Performance Summary — All Images",
                 fontsize=12, fontweight="bold", color=PAL["text0"])

    for ax, met in zip(axes.flat, metrics):
        vals = df[met].values
        bars = ax.bar(short, vals, color=colours, edgecolor=PAL["border"], lw=0.7)
        ax.set_title(met, fontsize=11, color=PAL["text0"])
        ax.set_ylim(0, min(max(vals) * 1.3, 1.1))
        mean_v = vals.mean()
        ax.axhline(mean_v, color="#F44336", ls="--", lw=1.5, label=f"Mean={mean_v:.3f}")
        ax.legend(fontsize=9); ax.set_facecolor(PAL["bg3"])
        ax.tick_params(colors=PAL["text1"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f"{v:.4f}", ha="center", fontsize=9, fontweight="bold", color=PAL["text0"])

    fig.tight_layout()
    return fig


def create_runtime_figure(df: pd.DataFrame) -> Figure:
    short   = [name.split("-")[1] + "\n" + name.split("-")[2][:5] for name in df["Image"]]
    colours = ["#42A5F5", "#66BB6A", "#FF9800", "#E91E63"]
    vals    = df["Running Time"].values

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Running Time per Image", fontsize=11, fontweight="bold", color=PAL["text0"])
    bars = ax.bar(short, vals, color=colours, edgecolor=PAL["border"], lw=0.7)
    ax.set_ylabel("Time (s)"); ax.set_ylim(0, max(vals) * 1.3)
    ax.axhline(vals.mean(), color="#F44336", ls="--", lw=1.5, label=f"Mean={vals.mean():.3f}s")
    ax.legend(fontsize=9); ax.set_facecolor(PAL["bg3"]); ax.tick_params(colors=PAL["text1"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f"{v:.3f}s", ha="center", fontsize=9, fontweight="bold", color=PAL["text0"])
    fig.tight_layout()
    return fig


def create_cross_step_figure(step_results: dict) -> Figure:
    if not step_results:
        return Figure()
    stages = list(step_results.values())[0]["stages"]
    n_s    = len(stages)
    x      = np.arange(n_s)
    width  = 0.20
    pal4   = ["#42A5F5", "#66BB6A", "#FF9800", "#E91E63"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Step-by-Step Performance — All Images", fontsize=12, fontweight="bold",
                 color=PAL["text0"])

    for idx, (name, sr) in enumerate(step_results.items()):
        sn  = name.split("-")[1]
        off = (idx - 1.5) * width
        axes[0].bar(x + off, sr["ious"],  width, color=pal4[idx], alpha=0.85,
                    edgecolor=PAL["border"], lw=0.5, label=sn)
        axes[1].bar(x + off, sr["dices"], width, color=pal4[idx], alpha=0.85,
                    edgecolor=PAL["border"], lw=0.5, label=sn)

    for ax, metric in zip(axes, ["IoU per Stage", "Dice per Stage"]):
        ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=7.5, color=PAL["text1"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title(metric, color=PAL["text0"])
        ax.legend(fontsize=8, title="Image", title_fontsize=8)
        ax.tick_params(colors=PAL["text1"]); ax.set_facecolor(PAL["bg3"])
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def create_cross_timing_figure(timing_results: dict) -> Figure:
    if not timing_results:
        return Figure()
    stage_lbls = list(timing_results.values())[0]["stages"]
    xt   = np.arange(len(stage_lbls))
    wt   = 0.20
    pal4 = ["#42A5F5", "#66BB6A", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Per-Stage Timing — All Images", fontsize=12, fontweight="bold", color=PAL["text0"])
    for idx, (name, tr) in enumerate(timing_results.items()):
        sn  = name.split("-")[1]
        off = (idx - 1.5) * wt
        ax.bar(xt + off, tr["times_ms"], wt, color=pal4[idx], alpha=0.85,
               edgecolor=PAL["border"], lw=0.5, label=sn)
    ax.set_xticks(xt); ax.set_xticklabels(stage_lbls, fontsize=8, color=PAL["text1"])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-Stage Time (ms, mean of 3 runs)", color=PAL["text0"])
    ax.legend(fontsize=9, title="Image"); ax.tick_params(colors=PAL["text1"])
    ax.set_facecolor(PAL["bg3"]); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
