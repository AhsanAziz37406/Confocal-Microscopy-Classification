import torch
from data_loader import get_data_loaders
from model import get_model
from test import evaluate_model
import os

def main():
    # Parameters
    data_dir = "new_may_dataset"  # Path to your dataset
    img_size = 512
    batch_size = 32
    model_path = 'models/maxvit_epoch_2.pth'  # Update path if needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data (train and val are not needed here)
    _, _, test_loader, class_names = get_data_loaders(data_dir, img_size, batch_size)

    # Load model
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluate
    evaluate_model(model, test_loader, device, class_names)

if __name__ == "__main__":
    main()
