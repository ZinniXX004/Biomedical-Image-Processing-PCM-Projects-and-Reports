# core/detector.py

import cv2

class FaceEyeDetector:
    """Pipeline Stage C: Haar Cascade. Eye = just decoration."""
    
    def __init__(self, face_scale=1.1, face_neighbors=5, face_min=(80, 80),
                 eye_scale=1.1, eye_neighbors=10, eye_min=(20, 20)):
        
        self.fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.ec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        self.fp = (face_scale, face_neighbors, face_min)
        self.ep = (eye_scale, eye_neighbors, eye_min)
        print("[FaceEyeDetector] Haar Cascade dimuat.")

    def detect(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.fc.detectMultiScale(gray, *self.fp[:2], minSize=self.fp[2])
        
        res = {'faces':[], 'eyes': {}, 'gray': gray}
        if len(faces) == 0:
            return res
            
        for idx, (fx, fy, fw, fh) in enumerate(faces):
            res['faces'].append((fx, fy, fw, fh))
            # Eye ROI only taken from upper part of face (top 60%)
            roi_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
            eyes = self.ec.detectMultiScale(roi_gray, *self.ep[:2], minSize=self.ep[2])
            res['eyes'][idx] =[(fx + ex, fy + ey, ew, eh) for ex, ey, ew, eh in eyes]
            
        return res

    def draw(self, frame, det):
        vis = frame.copy()
        for idx, (fx, fy, fw, fh) in enumerate(det['faces']):
            cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(vis, f"Face #{idx+1}", (fx, fy - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            
            for ex, ey, ew, eh in det['eyes'].get(idx,[]):
                cv2.rectangle(vis, (ex, ey), (ex + ew, ey + eh), (255, 100, 0), 2)
                
        return vis