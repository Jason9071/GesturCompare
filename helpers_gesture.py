import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, max_num_hands=1, detection_confidence=0.6, tracking_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,            # 單張圖片較穩
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.drawer = mp.solutions.drawing_utils

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        pts = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            h, w, _ = image.shape
            # 用「浮點像素座標」避免整數除法誤差
            pts = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark], dtype=np.float32)
            # 畫關鍵點
            self.drawer.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return pts, image


class GestureRules:
    @staticmethod
    def classify(pts):
        """
        規則法：THUMB_UP / V / OK / UNKNOWN
        - 以「垂直方向」判斷拇指是否向上，與左右手無關
        - 其他四指允許最多 1 指稍微伸直（更寬鬆）
        """
        if pts is None:
            return "UNKNOWN"

        # 手掌尺度（避免遠近差異）
        wrist = pts[0]
        palm_scale = np.linalg.norm(pts[9] - wrist) + 1e-6  # wrist -> middle_mcp

        def extended_vertical(tip_idx, pip_idx, thr=0.08):
            """ tip 在 pip 上方多少（y 越小越上方），以 palm_scale 正規化 """
            tip, pip = pts[tip_idx], pts[pip_idx]
            return (pip[1] - tip[1]) / palm_scale > thr

        # 各指是否伸直（以垂直方向為主）
        thumb_up    = (wrist[1] - pts[4][1]) / palm_scale > 0.12   # 👍 拇指是否「向上」(比手腕更高)
        index_ext   = extended_vertical(8, 6, 0.08)
        middle_ext  = extended_vertical(12,10, 0.08)
        ring_ext    = extended_vertical(16,14, 0.08)
        pinky_ext   = extended_vertical(20,18, 0.08)

        # OK：拇指尖與食指尖距離很近（圓圈）
        ok_close = np.linalg.norm(pts[4] - pts[8]) / palm_scale < 0.18

        # ---- 規則判斷順序：OK -> THUMB_UP -> V ----
        if ok_close:
            return "OK"

        # 👍：拇指明顯「向上」，其他四指「大多未伸直」（允許最多 1 指稍伸）
        if thumb_up:
            others_extended = sum([index_ext, middle_ext, ring_ext, pinky_ext])
            if others_extended <= 1:
                return "THUMB_UP"

        # ✌️：食指與中指伸直，無名指/小指未伸直
        if index_ext and middle_ext and not (ring_ext or pinky_ext):
            return "V"

        return "UNKNOWN"
