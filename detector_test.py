import argparse
import os

import cv2
import numpy as np

from megadetector import MegaDetector
from megadetector_v6 import MegaDetectorV6
from sam_detector import SAMDetector


CLASS_NAMES = {
    0: "animal",
    1: "person",
    2: "vehicle",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 165, 255),
    2: (255, 200, 0),
}


def summarize(class_ids: np.ndarray) -> dict:
    out = {"animal": 0, "person": 0, "vehicle": 0, "total": 0}
    for cid in class_ids:
        cid = int(cid)
        if cid in CLASS_NAMES:
            out[CLASS_NAMES[cid]] += 1
            out["total"] += 1
    return out


def draw(image_bgr, scores, class_ids, boxes, title):
    vis = image_bgr.copy()

    for score, cid, box in zip(scores, class_ids, boxes):
        cid = int(cid)
        if cid not in CLASS_NAMES:
            continue

        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS[cid]
        label = f"{CLASS_NAMES[cid]} {float(score):.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(vis, title, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return vis


def compare(image_path: str, conf: float, output_dir: str,
            md5, md6, samd):

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return

    s5, c5, b5 = md5(img)
    s6, c6, b6 = md6(img)
    ss, cs, bs = samd(img)

    sum5 = summarize(c5)
    sum6 = summarize(c6)
    sums = summarize(cs)

    vis5 = draw(img, s5, c5, b5,
                f"MDv5 total={sum5['total']} animals={sum5['animal']}")

    vis6 = draw(img, s6, c6, b6,
                f"MDv6 total={sum6['total']} animals={sum6['animal']}")

    viss = draw(img, ss, cs, bs,
                f"SAM total={sums['total']} animals={sums['animal']}")

    combined = np.hstack([vis5, vis6, viss])

    scale = 0.4
    combined_small = cv2.resize(combined, (0, 0), fx=scale, fy=scale)

    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)

    out_path = os.path.join(output_dir, f"{name}_comparison.jpg")

    cv2.imwrite(out_path, combined_small)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",
                        default="../small_example/adk01/transects/transect1/detection_frames/")
    parser.add_argument("--conf", type=float, default=0.2)

    args = parser.parse_args()

    if not os.path.exists(args.folder):
        raise FileNotFoundError(args.folder)

    output_dir = os.path.join(
         "/detection_results")
    os.makedirs(output_dir, exist_ok=True)

    # Load models ONCE (important for speed)
    md5 = MegaDetector(conf=args.conf)
    md6 = MegaDetectorV6(conf=args.conf)
    samd = SAMDetector(conf=args.conf)

    for filename in sorted(os.listdir(args.folder)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(args.folder, filename)

        compare(image_path, args.conf, output_dir,
                md5, md6, samd)


if __name__ == "__main__":
    main()