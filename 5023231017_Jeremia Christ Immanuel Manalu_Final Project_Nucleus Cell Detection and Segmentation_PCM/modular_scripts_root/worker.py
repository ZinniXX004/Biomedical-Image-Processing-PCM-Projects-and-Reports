"""Background QThread that drives the full segmentation pipeline"""

import time
import traceback
from copy import deepcopy
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from .config import DEFAULT_IMAGE_NAMES, IMG_SHORT
from .core   import (
    load_image, parse_xml_to_mask,
    get_h_channel,
    segment_nuclei, compute_metrics,
    collect_stepwise_data, collect_timing_data,
)

try:
    from scipy import ndimage as ndi
except ImportError:
    import scipy.ndimage as ndi


class ProcessingWorker(QThread):
    log           = pyqtSignal(str)
    progress      = pyqtSignal(int, int, str)

    hist_ready    = pyqtSignal(str, object, object)
    clahe_ready   = pyqtSignal(str, object, dict)
    seg_ready     = pyqtSignal(str, object, object, object, dict)
    step_ready    = pyqtSignal(str, dict)
    timing_ready  = pyqtSignal(str, dict)

    all_done      = pyqtSignal(list)
    error         = pyqtSignal(str)
    finished_     = pyqtSignal()

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg    = cfg
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        cfg      = self.cfg
        names    = cfg["image_names"]
        base_dir = Path(cfg["base_dir"])
        results  = []

        for i, name in enumerate(names):
            if self._abort:
                self.log.emit("⚠  Aborted by user.")
                break
            self.progress.emit(i, len(names), name)
            self.log.emit(
                f"\n{'═'*56}\n  Processing: {IMG_SHORT.get(name, name)}\n{'═'*56}")

            img_path = base_dir / "Tissue Images" / f"{name}.tif"
            xml_path = base_dir / "Annotations"   / f"{name}.xml"

            try:
                image = load_image(img_path)
                self.log.emit(f"  ✓ Image loaded  {image.shape}  {image.dtype}")
            except Exception as e:
                self.log.emit(f"  ✗ Cannot load image: {e}")
                self.error.emit(f"Cannot load image for {name}: {e}")
                continue

            try:
                gt_mask = parse_xml_to_mask(xml_path, image.shape)
                n_nuc   = ndi.label(gt_mask > 0)[1]
                self.log.emit(f"  ✓ Ground truth parsed  ({n_nuc} nuclei)")
            except Exception as e:
                self.log.emit(f"  ✗ Cannot parse XML: {e}")
                self.error.emit(f"Cannot parse annotation for {name}: {e}")
                continue

            try:
                H_u8_raw, _ = get_h_channel(image, mode=cfg["stain_mode"])
                self.hist_ready.emit(name, image.copy(), H_u8_raw.copy())
                self.log.emit("  ✓ Histogram data ready")
            except Exception as e:
                self.log.emit(f"  ✗ Histogram error: {e}")

            try:
                self.clahe_ready.emit(name, H_u8_raw.copy(), deepcopy(cfg["params"]))
                self.log.emit("  ✓ CLAHE diagnostic data ready")
            except Exception as e:
                self.log.emit(f"  ✗ CLAHE error: {e}")

            if cfg.get("run_diagnostics", True):
                try:
                    step_data = collect_stepwise_data(
                        image, gt_mask, name, cfg["params"],
                        cfg["threshold_map"], cfg["percentile_map"], cfg["stain_mode"])
                    self.step_ready.emit(name, step_data)
                    self.log.emit("  ✓ Step-wise analysis complete")
                except Exception as e:
                    self.log.emit(
                        f"  ✗ Step analysis error: {e}\n{traceback.format_exc()}")

            try:
                t0        = time.perf_counter()
                pred_mask = segment_nuclei(
                    image, name, cfg["params"], False,
                    cfg["use_watershed"],
                    cfg["threshold_map"], cfg["percentile_map"], cfg["stain_mode"])
                elapsed = time.perf_counter() - t0
                metrics = compute_metrics(pred_mask, gt_mask)
                self.log.emit(
                    f"  ✓ Segmentation done in {elapsed:.3f}s\n"
                    f"     IoU={metrics['IoU']:.4f}  Dice={metrics['Dice']:.4f}")
                self.seg_ready.emit(
                    name, image.copy(), gt_mask.copy(), pred_mask.copy(), dict(metrics))
            except Exception as e:
                elapsed = 0.0
                metrics = {"IoU": 0, "Dice": 0, "Precision": 0, "Recall": 0,
                           "TP": 0, "FP": 0, "FN": 0, "TN": 0}
                self.log.emit(f"  ✗ Segmentation error: {e}\n{traceback.format_exc()}")
                self.error.emit(f"Segmentation failed for {name}: {e}")
                continue

            try:
                timing_data = collect_timing_data(
                    image, name, cfg["params"],
                    cfg["threshold_map"], cfg["percentile_map"],
                    cfg["stain_mode"], n_repeats=cfg.get("timing_repeats", 3))
                self.timing_ready.emit(name, timing_data)
                self.log.emit(
                    f"  ✓ Timing profiled  total≈{timing_data['total_ms']:.1f}ms")
            except Exception as e:
                self.log.emit(f"  ✗ Timing error: {e}")
                timing_data = {}

            results.append({
                "Image":        name,
                "IoU":          round(metrics["IoU"],       4),
                "Dice":         round(metrics["Dice"],      4),
                "Precision":    round(metrics["Precision"], 4),
                "Recall":       round(metrics["Recall"],    4),
                "Running Time": round(elapsed,              4),
                "Threshold":    cfg["threshold_map"].get(name, "otsu"),
                "Pct":          cfg["percentile_map"].get(name, 0.0),
                "Device":       "GPU" if _gpu_available() else "CPU",
            })

        self.progress.emit(len(names), len(names), "Complete")
        self.all_done.emit(results)
        self.finished_.emit()
        self.log.emit("\n✅  All images processed.")


def _gpu_available() -> bool:
    try:
        import cupy as cp
        cp.zeros(1)
        return True
    except Exception:
        return False
