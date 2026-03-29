#!/usr/bin/env python3
"""
Standalone distance estimation script.
Processes a directory of images using SAM detector + SAM3 segmentation + depth model.
Saves results to results.csv in the image directory.
"""

from typing import Literal
import argparse
import os
import glob
import logging
import math
import csv
import numpy as np
import cv2

from config import Config
from dpt_pytorch import DPTPyTorch
from depth_anything import DepthAnything
from depth_anything_v3_pytorch import Depth_Anything_3
from sam_detector import SAMDetector, SAMDetectorLabel
from sam3_wrapper import SAM3
from de_utils import crop, resize, imread, exception_to_str


def _squeeze_single_channel(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Standalone distance estimation on images without calibration."
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--depth_model",
        type=str,
        choices=["dpt_pytorch", "depth_anything", "depth_anything_3"],
        default="depth_anything_3",
        help="Depth estimation model to use",
    )
    parser.add_argument(
        "--sam_detector_prompt",
        type=str,
        default="Sign Being Held",
        help="Text prompt for SAM detector",
    )
    parser.add_argument(
        "--bbox_confidence_threshold",
        type=float,
        default=0.2,
        help="Minimum confidence threshold for detections",
    )
    parser.add_argument(
        "--camera_horizontal_fov",
        type=float,
        default=40,
        help="Horizontal field of view in degrees",
    )
    parser.add_argument(
        "--camera_vertical_fov",
        type=float,
        default=30,
        help="Vertical field of view in degrees",
    )
    parser.add_argument(
        "--crop_top",
        type=int,
        default=0,
        help="Pixels to crop from top",
    )
    parser.add_argument(
        "--crop_bottom",
        type=int,
        default=0,
        help="Pixels to crop from bottom",
    )
    parser.add_argument(
        "--crop_left",
        type=int,
        default=0,
        help="Pixels to crop from left",
    )
    parser.add_argument(
        "--crop_right",
        type=int,
        default=0,
        help="Pixels to crop from right",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.5,
        help="Minimum depth value",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=25.0,
        help="Maximum depth value",
    )
    parser.add_argument(
        "--intensity_extensions",
        type=str,
        nargs="+",
        default=[".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG"],
        help="Image file extensions to process",
    )
    parser.add_argument(
        "--detect_humans",
        action="store_true",
        help="Include human detections",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Validate image directory
    if not os.path.isdir(args.image_dir):
        logging.error(f"Image directory does not exist: {args.image_dir}")
        return

    # Initialize depth model
    logging.info(f"Loading depth model: {args.depth_model}")
    if args.depth_model == "dpt_pytorch":
        depth_model = DPTPyTorch()
    elif args.depth_model == "depth_anything":
        depth_model = DepthAnything()
    elif args.depth_model == "depth_anything_3":
        depth_model = Depth_Anything_3()
    else:
        logging.error(f"Unknown depth model: {args.depth_model}")
        return

    # Initialize SAM detector
    logging.info(f"Loading SAM detector with prompt: '{args.sam_detector_prompt}'")
    sam_detector = SAMDetector(
        prompt=args.sam_detector_prompt,
        conf=args.bbox_confidence_threshold,
    )

    # Initialize SAM3 segmentor
    logging.info("Loading SAM3 segmentor")
    sam3_model = SAM3()

    # Collect images
    image_filenames = []
    for ext in args.intensity_extensions:
        image_filenames.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
    image_filenames = sorted(list(set(image_filenames)))

    if not image_filenames:
        logging.warning(f"No images found in {args.image_dir}")
        return

    logging.info(f"Found {len(image_filenames)} images to process")

    # Prepare results file
    results_csv_path = os.path.join(args.image_dir, "results.csv")
    results_txt_path = os.path.join(args.image_dir, "results.txt")

    bbox_result_columns = [
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "bbox_area",
        "bbox_area_percent",
    ]
    head_row_csv = [
        "frame_id",
        "detection_idx",
        "detection_confidence",
        "depth",
        "world_x",
        "world_y",
        "world_z",
    ] + bbox_result_columns + ["error_status"]

    head_row_txt = ["Frame*Detection", "Observation*Radial distance"]

    with open(results_csv_path, "w", newline="") as csv_file, open(
        results_txt_path, "w"
    ) as txt_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(head_row_csv)
        txt_file.write("\t".join(head_row_txt) + os.linesep)

        # Process images
        for img_idx, image_filename in enumerate(image_filenames):
            frame_id = os.path.splitext(os.path.basename(image_filename))[0]
            logging.info(f"[{img_idx + 1}/{len(image_filenames)}] Processing: {frame_id}")

            try:
                # Load and preprocess image
                img = imread(image_filename)
                img = crop(
                    img,
                    args.crop_top,
                    args.crop_bottom,
                    args.crop_left,
                    args.crop_right,
                )

                if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                    raise RuntimeError(
                        f"Image became empty after cropping "
                        f"(top={args.crop_top}, bottom={args.crop_bottom}, "
                        f"left={args.crop_left}, right={args.crop_right})."
                    )

                # Run SAM detector
                scores, labels, boxes = sam_detector(img)

                # Filter detections
                if args.detect_humans:
                    correct_label_idx = np.nonzero(
                        (labels.flatten() == SAMDetectorLabel.ANIMAL)
                        | (labels.flatten() == SAMDetectorLabel.PERSON)
                    )
                else:
                    correct_label_idx = np.nonzero(
                        labels.flatten() == SAMDetectorLabel.ANIMAL
                    )
                scores, labels, boxes = (
                    scores[correct_label_idx],
                    labels[correct_label_idx],
                    boxes[correct_label_idx],
                )

                # Filter by confidence
                high_confidence_idx = np.nonzero(
                    scores.flatten() >= args.bbox_confidence_threshold
                )
                scores, labels, boxes = (
                    scores[high_confidence_idx],
                    labels[high_confidence_idx],
                    boxes[high_confidence_idx],
                )

                # Sort from center outwards
                centerness = [
                    ((img.shape[1] / 2) - (box[0] + box[2]) / 2) ** 2
                    + ((img.shape[0] / 2) - (box[1] + box[3]) / 2) ** 2
                    for box in boxes
                ]
                centerness_idx = np.argsort(centerness)
                scores, labels, boxes = (
                    scores[centerness_idx],
                    labels[centerness_idx],
                    boxes[centerness_idx],
                )

                # Get SAM3 masks
                masks = sam3_model(img, boxes)

                # Estimate depth using selected model
                depth = depth_model(img)
                depth = _squeeze_single_channel(depth)
                depth = np.clip(depth, args.min_depth, args.max_depth)

                # Compute bbox attributes
                frame_h, frame_w = img.shape[0], img.shape[1]
                frame_area = float(max(1, frame_h * frame_w))
                bbox_attributes = []
                for i, box in enumerate(boxes):
                    xmin_f = max(0.0, min(float(frame_w), float(box[0])))
                    ymin_f = max(0.0, min(float(frame_h), float(box[1])))
                    xmax_f = max(0.0, min(float(frame_w), float(box[2])))
                    ymax_f = max(0.0, min(float(frame_h), float(box[3])))
                    width = max(0.0, xmax_f - xmin_f)
                    height = max(0.0, ymax_f - ymin_f)
                    bbox_area = width * height
                    bbox_area_percent = (100.0 * width * height) / frame_area
                    attrs = [
                        f"{xmin_f:.2f}",
                        f"{ymin_f:.2f}",
                        f"{xmax_f:.2f}",
                        f"{ymax_f:.2f}",
                        f"{bbox_area:.2f}",
                        f"{bbox_area_percent:.4f}",
                    ]
                    bbox_attributes.append(attrs)

                # Sample depths and compute world positions
                for box_idx, (score, box, mask) in enumerate(zip(scores, boxes, masks)):
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue

                    # Get mask region
                    ymin, ymax = (
                        max(0, min(depth.shape[0] - 2, round(box[1]))),
                        max(0, min(depth.shape[0] - 1, round(box[3]))),
                    )
                    xmin, xmax = (
                        max(0, min(depth.shape[1] - 2, round(box[0]))),
                        max(0, min(depth.shape[1] - 1, round(box[2]))),
                    )
                    depth_cropped = depth[ymin:ymax, xmin:xmax]
                    mask_cropped = mask[ymin:ymax, xmin:xmax]

                    # Find center of mask using distance transform
                    mask_padded = np.pad(mask_cropped, ((1, 1), (1, 1)))
                    dist = cv2.distanceTransform(
                        (mask_padded * 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3
                    )
                    sample_location = np.unravel_index(
                        np.argmax(dist, axis=None), dist.shape
                    )
                    sample_location = (
                        max(
                            0,
                            min(
                                mask_cropped.shape[0],
                                sample_location[0] - 1,
                            ),
                        ),
                        max(
                            0,
                            min(
                                mask_cropped.shape[1],
                                sample_location[1] - 1,
                            ),
                        ),
                    )

                    sampled_depth = depth_cropped[
                        sample_location[0], sample_location[1]
                    ]
                    sample_location_img = (
                        round(sample_location[0] + box[1]),
                        round(sample_location[1] + box[0]),
                    )

                    # Compute horizontal angle a
                    f = (0.5 * depth.shape[1]) / math.tan(
                        0.5 * math.pi * args.camera_horizontal_fov / 180
                    )
                    c = np.array([0, 0, f])
                    p = np.array(
                        [
                            (box[0] + box[2]) / 2 - depth.shape[1] / 2,
                            0,
                            f,
                        ]
                    )
                    a = math.copysign(1, (box[0] + box[2]) / 2 - depth.shape[1] / 2) * math.acos(
                        (c @ p) / (np.linalg.norm(c) * np.linalg.norm(p))
                    )

                    # Compute vertical angle b
                    f = (0.5 * depth.shape[0]) / math.tan(
                        0.5 * math.pi * args.camera_vertical_fov / 180
                    )
                    c = np.array([0, 0, f])
                    p = np.array(
                        [
                            0,
                            (box[1] + box[3]) / 2 - depth.shape[0] / 2,
                            f,
                        ]
                    )
                    b = math.copysign(1, (box[1] + box[3]) / 2 - depth.shape[0] / 2) * math.acos(
                        (c @ p) / (np.linalg.norm(c) * np.linalg.norm(p))
                    )

                    # Compute world position
                    z = sampled_depth / math.sqrt(
                        math.tan(a) ** 2 + math.tan(b) ** 2 + 1
                    )
                    x = z * math.tan(a)
                    y = z * math.tan(b)

                    # Write results
                    score_value = (
                        score.item() if hasattr(score, "item") else float(score)
                    )
                    sampled_depth_value = (
                        sampled_depth.item()
                        if hasattr(sampled_depth, "item")
                        else float(sampled_depth)
                    )
                    world_x = (
                        x.item() if hasattr(x, "item") else float(x)
                    )
                    world_y = (
                        y.item() if hasattr(y, "item") else float(y)
                    )
                    world_z = (
                        z.item() if hasattr(z, "item") else float(z)
                    )

                    row = [
                        frame_id,
                        f"{box_idx:03d}",
                        f"{score_value:.4f}",
                        f"{sampled_depth_value:.4f}",
                        f"{world_x:.4f}",
                        f"{world_y:.4f}",
                        f"{world_z:.4f}",
                    ] + bbox_attributes[box_idx] + [""]

                    with open(results_csv_path, "a", newline="") as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(row)

                    with open(results_txt_path, "a") as txt_file:
                        txt_file.write(
                            f"{frame_id}_{box_idx:03d}\t{sampled_depth_value:.4f}\n"
                        )

            except Exception as e:
                exception_str = exception_to_str(e)
                logging.error(f"Error processing '{frame_id}': {exception_str}")
                with open(results_csv_path, "a", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(
                        [
                            frame_id,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            exception_str,
                        ]
                    )

    logging.info(f"Results saved to {results_csv_path} and {results_txt_path}")


if __name__ == "__main__":
    main()
