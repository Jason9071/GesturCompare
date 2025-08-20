# Gesture-Only Verification (Local PoC)

僅以「手勢」驗證，不做任何人臉/樣貌比對。

-   手部關鍵點：MediaPipe Hands（21 點）
-   判斷方式：規則法 (THUMB_UP / V / OK)；可選擇升級為小型分類器（scikit-learn）

## 0) 安裝

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

python verify_gesture.py data\verify\test6.jpeg OK
python verify_gesture.py data\verify\test6.jpeg THUMB_UP
python verify_gesture.py data\verify\test.jpeg V
