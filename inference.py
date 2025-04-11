import os
import argparse
import torch
import json
import pandas as pd
from tqdm import tqdm
from model import get_model
from utils import get_test_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./data/test', help='Path to test image directory')
    parser.add_argument('--model_type', type=str, default='resnext50_32x4d', help='Path to test image directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth')
    parser.add_argument('--output_dir', type=str, default='./result', help='Directory to save pred.json and pred.csv')
    parser.add_argument('--threshold', type=float, default=0.3, help='Score threshold for filtering predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def run_inference(args):
    device = torch.device(args.device)

    # Load model
    model = get_model(model_tpye=args.model_type, weights=args.model_path)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    test_loader = get_test_loader()

    task1_output = []  # For pred.json
    task2_output = []  # For pred.csv

    for images, image_ids in tqdm(test_loader):
        images = [img.to(device) for img in images]
        image_id = image_ids[0]

        with torch.no_grad():
            outputs = model(images)[0]

        boxes = outputs["boxes"].cpu()
        scores = outputs["scores"].cpu()
        labels = outputs["labels"].cpu()

        # Task 1: Generate COCO-format predictions
        for box, score, label in zip(boxes, scores, labels):
            if score < args.threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            task1_output.append({
                "image_id": int(image_id),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "category_id": int(label)
            })

        # Task 2: Whole image prediction by sorting digits left to right
        keep = scores > args.threshold
        boxes = boxes[keep]
        labels = labels[keep]
        if len(boxes) == 0:
            pred_label = "-1"
        else:
            x_coords = boxes[:, 0]
            sorted_indices = x_coords.argsort()
            sorted_labels = labels[sorted_indices]
            pred_label = ''.join(str(l.item()) for l in sorted_labels)

        task2_output.append({
            "image_id": int(image_id),
            "pred_label": pred_label
        })

    # Save results
    os.makedirs(f'{args.output_dir}', exist_ok=True)

    json_path = os.path.join(args.output_dir, "pred.json")
    with open(json_path, "w") as f:
        json.dump(task1_output, f)
    print(f"Task1 (Detection) saved to {json_path}")

    csv_path = os.path.join(args.output_dir, "pred.csv")
    pd.DataFrame(task2_output).to_csv(csv_path, index=False)
    print(f"Task2 (Recognition) saved to {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
