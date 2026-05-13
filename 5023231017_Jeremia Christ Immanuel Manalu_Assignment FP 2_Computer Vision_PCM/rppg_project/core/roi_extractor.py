# core/roi_extractor.py

import cv2
import time
import numpy as np
from collections import deque

class MultiROIExtractor:
    """Pipeline Stage D and E: ROI Selection and Signal Extraction."""
    
    PRESETS = {
        'forehead':    (0.25, 0.05, 0.50, 0.20),
        'cheek_left':  (0.05, 0.45, 0.35, 0.25),
        'cheek_right': (0.60, 0.45, 0.35, 0.25)
    }
    
    ROI_COLORS = {
        'forehead':    (0, 0, 255),
        'cheek_left':  (255, 100, 0),
        'cheek_right': (255, 200, 0)
    }

    def __init__(self, buffer_seconds=10.0, fps=30.0):
        self.fps = fps
        self.buffer_len = int(buffer_seconds * fps)
        self.buffers = {}
        
        for mode in ('A_forehead', 'B_cheek', 'C_combined'):
            self.buffers[mode] = {k: deque(maxlen=self.buffer_len) for k in ('R', 'G', 'B', 'time')}
            
        print(f"[MultiROIExtractor] buffer={buffer_seconds}s x {fps}fps = {self.buffer_len} samples/mode")

    def _coords(self, face_bbox, roi_name):
        fx, fy, fw, fh = face_bbox
        xf, yf, wf, hf = self.PRESETS[roi_name]
        return (
            int(fx + xf * fw), 
            int(fy + yf * fh), 
            max(int(wf * fw), 10), 
            max(int(hf * fh), 10)
        )

    def _patch_mean(self, frame, face_bbox, roi_name):
        h, w = frame.shape[:2]
        rx, ry, rw, rh = self._coords(face_bbox, roi_name)
        patch = frame[ry:min(ry + rh, h), rx:min(rx + rw, w)]
        return patch.mean(axis=(0, 1)) if patch.size > 0 else None

    def update(self, frame, face_bbox, timestamp=None):
        if timestamp is None: 
            timestamp = time.time()
            
        fh = self._patch_mean(frame, face_bbox, 'forehead')
        cl = self._patch_mean(frame, face_bbox, 'cheek_left')
        cr = self._patch_mean(frame, face_bbox, 'cheek_right')
        
        if fh is None and cl is None and cr is None: 
            return
            
        if fh is not None: 
            self._push('A_forehead', fh, timestamp)
            
        vck =[x for x in [cl, cr] if x is not None]
        if vck: 
            self._push('B_cheek', np.mean(vck, axis=0), timestamp)
            
        all_r = [x for x in [fh, cl, cr] if x is not None]
        if all_r:
            self._push('C_combined', np.mean(all_r, axis=0), timestamp)

    def _push(self, mode, bgr, ts):
        b = self.buffers[mode]
        b['B'].append(bgr[0])
        b['G'].append(bgr[1])
        b['R'].append(bgr[2])
        b['time'].append(ts)

    def get_signals(self, mode):
        b = self.buffers[mode]
        return {k: np.array(b[k], dtype=np.float64) for k in ('R', 'G', 'B', 'time')} | {'n_samples': len(b['R'])}

    def is_ready(self, mode, min_sec=5.0):
        return len(self.buffers[mode]['R']) >= int(min_sec * self.fps)

    def reset(self):
        for m in self.buffers:
            for k in self.buffers[m]: 
                self.buffers[m][k].clear()
        print("[MultiROIExtractor] All buffers reset.")

    def draw(self, frame, face_bbox):
        h, w = frame.shape[:2]
        for name, color in self.ROI_COLORS.items():
            rx, ry, rw, rh = self._coords(face_bbox, name)
            cv2.rectangle(frame, (rx, ry), (min(rx + rw, w), min(ry + rh, h)), color, 2)
            lbl = {'forehead': 'FH', 'cheek_left': 'CL', 'cheek_right': 'CR'}[name]
            cv2.putText(frame, lbl, (rx + 3, ry + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        return frame