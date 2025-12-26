import cv2
from deepface import DeepFace
import threading
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
EMOTION_MAP = {
    "angry":    {"emoji": "😠", "color": (0, 0, 255)},
    "happy":    {"emoji": "😊", "color": (0, 255, 255)},
    "sad":      {"emoji": "😢", "color": (255, 50, 50)},
    "surprise": {"emoji": "😲", "color": (255, 0, 255)},
    "neutral":  {"emoji": "😐", "color": (0, 255, 0)}
}
EMOTIONS = list(EMOTION_MAP.keys())

class SuperEmotionHUD:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Set resolution to 720p or 1080p for better look
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.current_frame = None
        self.raw_scores = {e: 0.0 for e in EMOTIONS}
        self.smooth_scores = {e: 0.0 for e in EMOTIONS}
        self.top_emotion = "neutral"
        self.running = True
        
        # Load Font for Emojis (Ensure you have a ttf font that supports emojis, default Windows: seguiemj.ttf)
        # If on Linux/Mac, change path to a supporting font like 'Apple Color Emoji.ttc' or 'NotoColorEmoji.ttf'
        try:
            self.font = ImageFont.truetype("seguiemj.ttf", 40)
        except:
            self.font = ImageFont.load_default()

        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _ai_worker(self):
        while self.running:
            if self.current_frame is not None:
                try:
                    res = DeepFace.analyze(self.current_frame, actions=['emotion'], enforce_detection=False, silent=True)
                    emo_data = res[0]['emotion']
                    for e in EMOTIONS:
                        self.raw_scores[e] = float(emo_data[e])
                    self.top_emotion = max(EMOTIONS, key=lambda e: self.raw_scores[e])
                except: pass
            time.sleep(0.2)

    def cv2_to_pil(self, frame):
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def pil_to_cv2(self, pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_ui(self, frame):
        h, w, _ = frame.shape
        
        # 1. Smooth the scores (Linear Interpolation for 'Liquid' bars)
        for e in EMOTIONS:
            self.smooth_scores[e] += (self.raw_scores[e] - self.smooth_scores[e]) * 0.2

        # 2. Cinematic Vignette & Grid
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w, h), (10, 10, 10), -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        
        # 3. Side Data Panel
        panel_w = 300
        padding = 30
        for i, emo in enumerate(EMOTIONS):
            y_pos = 150 + i * 60
            score = self.smooth_scores[emo]
            color = EMOTION_MAP[emo]["color"]
            
            # Label
            cv2.putText(frame, emo.upper(), (padding, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            
            # Glow Bar
            bar_max = 150
            bar_val = int((score/100) * bar_max)
            cv2.rectangle(frame, (padding, y_pos+10), (padding+bar_max, y_pos+15), (30,30,30), -1)
            cv2.rectangle(frame, (padding, y_pos+10), (padding+bar_val, y_pos+15), color, -1)

        # 4. Emoji & Top Status (Using PIL for Emoji support)
        pil_img = self.cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_img)
        
        # Draw Big Emoji next to face or in corner
        current_emoji = EMOTION_MAP[self.top_emotion]["emoji"]
        draw.text((padding, 70), f"{current_emoji} {self.top_emotion.upper()}", font=self.font, fill=(255, 255, 255))
        
        frame = self.pil_to_cv2(pil_img)

        # 5. Face Tracking HUD
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, fw, fh) in faces:
            color = EMOTION_MAP[self.top_emotion]["color"]
            # Minimalist "Focus" Corners
            length = 40
            cv2.line(frame, (x, y), (x+length, y), color, 2)
            cv2.line(frame, (x, y), (x, y+length), color, 2)
            cv2.line(frame, (x+fw, y+fh), (x+fw-length, y+fh), color, 2)
            cv2.line(frame, (x+fw, y+fh), (x+fw, y+fh-length), color, 2)
            
            # ID Tag
            cv2.rectangle(frame, (x, y-40), (x+fw, y), color, -1)
            cv2.putText(frame, f"ANALYZING SUBJECT: {int(self.smooth_scores[self.top_emotion])}%", 
                        (x+5, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        return frame

    def run(self):
        window_name = "AI_EMOTION_HUD"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            ui_frame = self.draw_ui(frame)
            
            cv2.imshow(window_name, ui_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SuperEmotionHUD().run()
