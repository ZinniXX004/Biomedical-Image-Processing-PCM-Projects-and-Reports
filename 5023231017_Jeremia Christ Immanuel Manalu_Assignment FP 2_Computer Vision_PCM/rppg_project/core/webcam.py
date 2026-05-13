# core/webcam.py

import cv2

class WebcamCapture:
    """Pipeline Stage A and B: Webcam + Frame Acquisition."""
    
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.actual_fps = fps

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[WebcamCapture] ERROR: camera {self.camera_index}")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[WebcamCapture] {aw}x{ah} @ {self.actual_fps:.1f} FPS")
        
        return True

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            print("[WebcamCapture] Released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *a):
        self.release()