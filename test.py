import argparse
import os

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

from constants.config_const import CLASSES, IMAGE_SIZE, NUM_CLASSES, THRESHOLD
from constants.paths_const import MODEL_FILE, TESTS_IMAGE_PATH, TRAINED_MODEL_PATH
from utils.download_model import download_best_model
from utils.file_helpers import get_new_file


def get_args():
    parser = argparse.ArgumentParser(description="Test a CNN model")
    parser.add_argument(
        "--model", "-m", type=str, default=os.path.join(TRAINED_MODEL_PATH, MODEL_FILE)
    )
    parser.add_argument(
        "--test_images",
        "-i",
        type=str,
        default=os.path.join(TESTS_IMAGE_PATH, "img1.jpg"),
    )
    parser.add_argument("--size", "-s", type=int, default=IMAGE_SIZE)
    parser.add_argument("--threshold", "-t", type=float, default=THRESHOLD)
    parser.add_argument("--output_test", "-o", default=None, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = os.path.join(TRAINED_MODEL_PATH, MODEL_FILE)
    if args.model == model_path and not os.path.exists(model_path):
        download_best_model()

    # Load model
    model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=NUM_CLASSES)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # # Process the image as input for the model
    ori_image = cv2.imread(args.test_images)
    if ori_image is None:
        print(
            f"Error: Could not load image at '{args.test_images}'. Please check the file path and integrity."
        )
        exit(1)
    height, width, _ = ori_image.shape
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.size, args.size)) / 255
    image = np.transpose(image, (2, 0, 1))
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float()
    image = image.to(device)

    # Use model for detect
    with torch.no_grad():
        predictions = model(image)

    # Get the results from the model's prediction
    prediction = predictions[0]
    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]
    final_boxes = []
    final_labels = []
    final_scores = []

    # Showing results by threshold
    for b, l, s in zip(boxes, labels, scores):
        if s > args.threshold:
            final_boxes.append(b)
            final_labels.append(l)
            final_scores.append(s)
    for b, l, s in zip(final_boxes, final_labels, final_scores):
        x_min, y_min, x_max, y_max = b
        x_min = int(x_min / args.size * width)
        x_max = int(x_max / args.size * width)
        y_min = int(y_min / args.size * height)
        y_max = int(y_max / args.size * height)

        # Draw bbox
        ori_image = cv2.rectangle(
            ori_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
        )

        # Draw label
        ori_image = cv2.putText(
            ori_image,
            CLASSES[l],
            (x_min, y_min),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if not args.output_test:
        args.output_test = get_new_file("images")
    print("The result of the model is saved at: ", args.output_test)

    cv2.imwrite(args.output_test, ori_image)
    cv2.imshow("image", ori_image)
    cv2.waitKey(0)
