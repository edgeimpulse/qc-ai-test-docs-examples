import cv2
import numpy as np
from PIL import Image
import os, time, argparse
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description='OpenCV -> Python RGB frames')
parser.add_argument('--device-id', type=int, required=True, help='E.g. 2 for /dev/video2')
args, unknown = parser.parse_known_args()

def gst_style_crop_resize_224_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Emulates:
      videoconvert ! aspectratiocrop aspect-ratio=1/1 ! videoscale ! video/x-raw,format=RGB,width=224,height=224
    Input:  BGR image (e.g., cv2.imread(...)) of any size, e.g. 1920x1080
    Output: RGB image, shape (224, 224, 3)
    """
    h, w = img_bgr.shape[:2]

    # 1) aspectratiocrop aspect-ratio=1/1  (center crop to square)
    if w >= h:
        side = h
        x0 = (w - side) // 2
        y0 = 0
    else:
        side = w
        x0 = 0
        y0 = (h - side) // 2
    cropped = img_bgr[y0:y0 + side, x0:x0 + side]

    # 2) videoscale to 224x224 using bilinear (matches videoscale default)
    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)

    # 3) videoconvert to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb

# Request 1920x1080 capture
cap = cv2.VideoCapture(args.device_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Async saving of images...
executor = ThreadPoolExecutor(max_workers=1)
def save_png_rgb(arr, filename):
    tmp = f"/tmp/{filename}"
    Image.fromarray(arr, "RGB").save(tmp)
    os.makedirs('out', exist_ok=True)
    os.replace(tmp, f"out/{filename}")  # atomic on POSIX

while True:
    # wait until next frame is ready
    cap.grab()

    p0 = time.perf_counter()
    ret, frame = cap.retrieve()
    if not ret:
        print("Failed to retrieve frame")
        break

    FILE_NAME = 'opencv.png'

    img_rgb_224 = gst_style_crop_resize_224_rgb(frame)
    p1 = time.perf_counter()

    print(f'[OpenCV] {(p1-p0)*1000:.2f}ms.')

    # Save image to disk (asynchronously)
    executor.submit(save_png_rgb, img_rgb_224.copy(), FILE_NAME)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
executor.shutdown(wait=True)
