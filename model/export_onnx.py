"""
Converts a trained checkpoint (.pt) into an ONNX model for serving.

ONNX Runtime is used at serve time instead of full PyTorch so the backend
container is smaller and inference startup is faster - the backend never
needs torch installed.

Usage:
    python export_onnx.py --checkpoint ../backend/checkpoint.pt --out ../backend/model.onnx
"""

import argparse
import json

import torch
from train import CharCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = CharCNN(num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy_input = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )

    # Save the class list alongside the model so the backend doesn't need
    # to import anything from model/ at serve time.
    classes_path = args.out.rsplit(".", 1)[0] + "_classes.json"
    with open(classes_path, "w") as f:
        json.dump(ckpt["classes"], f)

    print(f"Exported ONNX model -> {args.out}")
    print(f"Exported class list -> {classes_path}")
    print(f"Checkpoint test accuracy was: {ckpt.get('test_acc'):.4f}")


if __name__ == "__main__":
    main()
