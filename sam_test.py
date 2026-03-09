import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sam import SAM
from sam3_wrapper import SAM3
from megadetector import MegaDetector, MegaDetectorLabel
from megadetector_v6 import MegaDetectorV6, MegaDetectorLabel as MegaDetectorV6Label


def parse_boxes(boxes_str):
    if boxes_str is None or boxes_str.strip() == "":
        return None

    boxes = []
    for chunk in boxes_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"Invalid box '{chunk}'. Expected format x1,y1,x2,y2 (use ';' between boxes)."
            )
        x1, y1, x2, y2 = [float(v) for v in parts]
        boxes.append([x1, y1, x2, y2])

    return np.array(boxes, dtype=np.float32)


def normalize_mask_to_image(mask, image_shape):
    target_h, target_w = image_shape[:2]
    mask_arr = np.asarray(mask)

    if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
        mask_arr = mask_arr[..., 0]
    elif mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr[0]

    if mask_arr.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {mask_arr.shape}")

    if mask_arr.shape == (target_h, target_w):
        return mask_arr.astype(bool)

    if mask_arr.shape == (target_w, target_h):
        return mask_arr.T.astype(bool)

    resized = cv2.resize(
        mask_arr.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def apply_mask_overlay(image_bgr, masks, color_bgr=(0, 255, 0), alpha=0.35):
    out = image_bgr.copy()
    if masks is None or len(masks) == 0:
        return out

    color_arr = np.array(color_bgr, dtype=np.float32)

    for mask in masks:
        mask_bool = normalize_mask_to_image(mask, out.shape)
        if not np.any(mask_bool):
            continue
        out[mask_bool] = (
            (1.0 - alpha) * out[mask_bool].astype(np.float32) + alpha * color_arr
        ).astype(np.uint8)

    return out


def draw_boxes(image_bgr, boxes, color_bgr=(0, 255, 255), thickness=2):
    out = image_bgr.copy()
    if boxes is None:
        return out

    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness)
    return out


def summarize_masks(masks):
    if masks is None or len(masks) == 0:
        return 0, 0.0
    mask_count = len(masks)
    areas = [float(np.asarray(mask).astype(bool).sum()) for mask in masks]
    return mask_count, float(np.mean(areas)) if areas else 0.0


def detect_pipeline_boxes(image, detector_name, conf_threshold, detect_humans):
    if detector_name == "megadetector_v6":
        detector = MegaDetectorV6()
        detector_labels = MegaDetectorV6Label
    else:
        detector = MegaDetector()
        detector_labels = MegaDetectorLabel

    scores, labels, boxes = detector(image)

    if detect_humans:
        keep_label_idx = np.nonzero(
            (labels.flatten() == detector_labels.ANIMAL)
            | (labels.flatten() == detector_labels.PERSON)
        )
    else:
        keep_label_idx = np.nonzero(labels.flatten() == detector_labels.ANIMAL)
    scores, labels, boxes = (
        scores[keep_label_idx],
        labels[keep_label_idx],
        boxes[keep_label_idx],
    )

    keep_conf_idx = np.nonzero(scores.flatten() >= conf_threshold)
    scores, labels, boxes = (
        scores[keep_conf_idx],
        labels[keep_conf_idx],
        boxes[keep_conf_idx],
    )

    centerness = [
        ((image.shape[1] / 2) - (box[0] + box[2]) / 2) ** 2
        + ((image.shape[0] / 2) - (box[1] + box[3]) / 2) ** 2
        for box in boxes
    ]
    center_idx = np.argsort(centerness)
    scores, labels, boxes = (
        scores[center_idx],
        labels[center_idx],
        boxes[center_idx],
    )

    return scores, labels, boxes


def run_and_visualize(
    image_path,
    sam_boxes,
    sam3_boxes,
    output_path,
    show,
    detector_name,
    conf_threshold,
    detect_humans,
):
    sam_img = cv2.imread(image_path)
    if sam_img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    sam3_img = sam_img.copy()

    md_scores, md_labels, md_boxes = detect_pipeline_boxes(
        sam_img,
        detector_name=detector_name,
        conf_threshold=conf_threshold,
        detect_humans=detect_humans,
    )

    sam_boxes = sam_boxes if sam_boxes is not None else md_boxes
    sam3_boxes = sam3_boxes if sam3_boxes is not None else md_boxes

    sam_model = SAM()
    sam3_model = SAM3()

    sam_masks = sam_model(sam_img, sam_boxes)
    sam3_masks = sam3_model(sam3_img, sam3_boxes)

    sam_vis = apply_mask_overlay(sam_img, sam_masks, color_bgr=(50, 220, 50), alpha=0.35)
    sam_vis = draw_boxes(sam_vis, sam_boxes, color_bgr=(0, 255, 255))

    sam3_vis = apply_mask_overlay(sam3_img, sam3_masks, color_bgr=(255, 100, 50), alpha=0.35)
    sam3_vis = draw_boxes(sam3_vis, sam3_boxes, color_bgr=(0, 255, 255))

    sam_count, sam_mean_area = summarize_masks(sam_masks)
    sam3_count, sam3_mean_area = summarize_masks(sam3_masks)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(cv2.cvtColor(sam_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"SAM (v1)\nMasks: {sam_count}, Mean area: {sam_mean_area:.1f}px")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(sam3_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"SAM3\nMasks: {sam3_count}, Mean area: {sam3_mean_area:.1f}px")
    axes[1].axis("off")

    plt.tight_layout()

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved comparison figure: {output_path}")

    print(f"Image: {image_path}")
    print(
        f"MegaDetector ({detector_name}) detections after filtering: {len(md_boxes)} "
        f"(threshold={conf_threshold}, detect_humans={detect_humans})"
    )
    if len(md_boxes) > 0:
        print(f"MegaDetector scores: {np.array2string(md_scores, precision=3)}")
        print(f"MegaDetector labels: {np.array2string(md_labels)}")
    print(f"SAM masks: {sam_count}, mean mask area: {sam_mean_area:.1f}px")
    print(f"SAM3 masks: {sam3_count}, mean mask area: {sam3_mean_area:.1f}px")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run MegaDetector then SAM and SAM3 on one image and compare results visually. "
            "Optional override box format: 'x1,y1,x2,y2;x1,y1,x2,y2'"
        )
    )
    parser.add_argument("image", help="Path to image used for both sam.py and sam3.py")
    parser.add_argument(
        "--sam-boxes",
        default="",
        help="Optional manual SAM boxes. If omitted, MegaDetector boxes are used.",
    )
    parser.add_argument(
        "--sam3-boxes",
        default="",
        help="Optional manual SAM3 boxes. If omitted, MegaDetector boxes are used.",
    )
    parser.add_argument(
        "--detector",
        choices=["megadetector", "megadetector_v6"],
        default="megadetector",
        help="Detection backend to generate boxes before SAM.",
    )
    parser.add_argument(
        "--bbox-threshold",
        type=float,
        default=0.2,
        help="Confidence threshold for MegaDetector boxes (matches main pipeline default).",
    )
    parser.add_argument(
        "--detect-humans",
        action="store_true",
        help="Keep person detections in addition to animals (matches main pipeline option).",
    )
    parser.add_argument(
        "--output",
        default="sam_vs_sam3_comparison.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib window",
    )

    args = parser.parse_args()

    sam_boxes = parse_boxes(args.sam_boxes)
    sam3_boxes = parse_boxes(args.sam3_boxes)

    run_and_visualize(
        image_path=args.image,
        sam_boxes=sam_boxes,
        sam3_boxes=sam3_boxes,
        output_path=args.output,
        show=not args.no_show,
        detector_name=args.detector,
        conf_threshold=args.bbox_threshold,
        detect_humans=args.detect_humans,
    )
