import torch
from data_loader import get_data_loaders
from model import get_model
from train import train_model
from test import evaluate_model
import os


def main():
    # Parameters
    data_dir = "new_may_dataset"  # Adjust path if needed
    img_size = 512
    batch_size = 4
    num_epochs = 2
    save_path = 'models'
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(data_dir, img_size, batch_size)

    # Model
    model = get_model(num_classes=len(class_names))

    # Train
    train_model(model, train_loader, val_loader, num_epochs, device, save_path)

    # Load best or last model for testing
    model.load_state_dict(torch.load(os.path.join(save_path, f'maxvit_epoch_{num_epochs}.pth')))

    # Test
    evaluate_model(model, test_loader, device, class_names)


if __name__ == "__main__":
    main()
