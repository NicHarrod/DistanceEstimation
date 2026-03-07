import argparse
import os

import cv2
import numpy as np

from megadetector import MegaDetector
from megadetector_v6 import MegaDetectorV6


CLASS_NAMES = {
    0: "animal",
    1: "person",
    2: "vehicle",
}

CLASS_COLORS = {
    0: (0, 255, 0),    # animal: green
    1: (0, 165, 255),  # person: orange
    2: (255, 200, 0),  # vehicle: cyan-ish
}


def summarize(class_ids: np.ndarray) -> dict:
    out = {"animal": 0, "person": 0, "vehicle": 0, "total": 0}
    for cid in class_ids:
        cid = int(cid)
        if cid in CLASS_NAMES:
            out[CLASS_NAMES[cid]] += 1
            out["total"] += 1
    return out


def draw(image_bgr: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, boxes: np.ndarray, title: str) -> np.ndarray:
    vis = image_bgr.copy()
    for score, cid, box in zip(scores, class_ids, boxes):
        cid = int(cid)
        if cid not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS[cid]
        label = f"{CLASS_NAMES[cid]} {float(score):.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(vis, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return vis

def compare(image_path: str, conf: float):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    md5 = MegaDetector(conf=conf)
    md6 = MegaDetectorV6(conf=conf)

    s5, c5, b5 = md5(img)
    s6, c6, b6 = md6(img)

    sum5 = summarize(c5)
    sum6 = summarize(c6)

    print("=== MegaDetector Comparison ===")
    print(f"Image: {image_path}")
    print(f"Conf: {conf}")
    print(f"MDv5: {sum5}")
    print(f"MDv6: {sum6}")

    vis5 = draw(img, s5, c5, b5, f"MDv5 total={sum5['total']} animals={sum5['animal']}")
    vis6 = draw(img, s6, c6, b6, f"MDv6 total={sum6['total']} animals={sum6['animal']}")

    combined = np.hstack([vis5, vis6])

    scale = 0.5  # 50% of original size
    combined_small = cv2.resize(combined, (0, 0), fx=scale, fy=scale)
    cv2.imshow("MDv5 (left) vs MDv6 (right)", combined_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main():
    parser = argparse.ArgumentParser(description="Compare MegaDetector v5 vs v6 on one image.")
    parser.add_argument("--folder", help="Path to sample folder", default="../small_example/adk18/transects/transect1/detection_frames/")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    for filename in os.listdir(args.folder):
        if not filename.endswith(".jpg"):
            continue
        image_path = os.path.join(args.folder, filename)
        compare(image_path, args.conf)
    


if __name__ == "__main__":
    main()