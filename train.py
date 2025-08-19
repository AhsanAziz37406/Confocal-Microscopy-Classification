import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


def train_model(model, train_loader, val_loader, num_epochs, device, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}] Loss: {running_loss/len(train_loader):.4f}")

        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(save_path, f'maxvit_epoch_{epoch+1}.pth'))

    print("Training complete.")
