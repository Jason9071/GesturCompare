import sys, cv2
from helpers_gesture import HandDetector, GestureRules

def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_gesture.py <image_path> <THUMB_UP|V|OK>")
        sys.exit(1)

    image_path = sys.argv[1]
    expected = sys.argv[2].upper()

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot read image {image_path}")
        sys.exit(1)

    detector = HandDetector()
    pts, debug_img = detector.detect(img)

    pred = "UNKNOWN"
    if pts is not None:
        pred = GestureRules.classify(pts)
        color = (0, 200, 0) if pred == expected else (0, 0, 255)
        cv2.putText(debug_img, f"PRED: {pred}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(debug_img, "No hand detected", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(debug_img, f"EXPECTED: {expected}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    out_path = "db/debug_gesture_result.jpg"
    cv2.imwrite(out_path, debug_img)
    print({"expected": expected, "pred": pred, "ok": pred == expected, "debug": out_path})

if __name__ == "__main__":
    main()
