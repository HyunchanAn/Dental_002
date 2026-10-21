import os
import torch
from ultralytics import YOLO


def train_refined():
    # Load the ALREADY TRAINED model as base for refinement
    model = YOLO('models/best.pt')
    device = 0 if torch.cuda.is_available() else 'cpu'

    print(f"Starting training on REFINED dataset (device={device})...")
    results = model.train(
        data='data_refined.yaml',
        epochs=100,
        imgsz=1024,
        batch=4,
        name='dentex_yolov11s_refined_v3',
        patience=20,
        save=True,
        device=device,
        fl_gamma=1.5  # Focal Loss (Issue #3)
    )

    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}")


if __name__ == '__main__':
    if not os.path.exists('data_refined.yaml'):
        print("Error: 'data_refined.yaml' not found.")
    else:
        train_refined()
