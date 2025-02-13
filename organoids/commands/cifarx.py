import os
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
from tqdm import tqdm
from PIL import Image

class ExtendedMNISTDataset(Dataset):
    def __init__(self, images, labels, image_size=48):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(), 
            transforms.Resize((image_size, image_size)), 
            transforms.RandomApply([
                transforms.RandomRotation(degrees=(-25, 25)),

            ], p=0.4),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]
    

class CifarXModel(nn.Module):
    def __init__(self, lr=None, wd=None, linear_layer_type: nn.Module = nn.Linear):
        super(CifarXModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.fc1 = linear_layer_type(1153 * 3 * 3, 512)
        self.fc1 = linear_layer_type(4608, 512)  # Adjusted input size
        self.fc2 = linear_layer_type(512, 13) # cls. 0-12
        self.lr = lr
        self.wd = wd

        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.conv3,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2
        )

        self.transform = transforms.Compose([
            transforms.Grayscale(), 
            transforms.Resize((48, 48)), 
            transforms.ToTensor()
        ])

        num_params = sum(p.numel() for p in self.model.parameters())
        print("NUM_PARAMS: ", num_params)

    def forward(self, x):
        return self.model(x)

    def classify(self, input_image):
       # Handle PIL Image
        if isinstance(input_image, Image.Image):
            # Transform PIL image to tensor
            x = self.transform(input_image)
            x = x.unsqueeze(0)  # Add batch dimension
        else:
            x = input_image
            # Add batch dimension if needed
            if x.dim() == 3:
                x = x.unsqueeze(0)
        
        # Set model to eval mode
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)

        # if single image returns scale otherwise not
        if predictions.size(0) == 1:
            return predictions.item()
        
        return predictions
        


def train_model(model, train_loader, test_loader, optimizer, criterion, epochs, device):
    # Move model to the specified device (CPU or GPU)
    model.to(device)
    
    # Training loop for n epochs
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Training loop
        for inputs, labels in tqdm(train_loader, desc="[TRAINING]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        total_accuracy = 0.0
        
        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, labels in tqdm(test_loader, desc="[VALID]"):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)  # Forward pass
                loss = criterion(logits, labels)  # Compute validation loss
                val_loss += loss.item()

                # Compute accuracy
                predictions = torch.argmax(logits, dim=-1)
                accuracy = predictions.eq(labels).float().mean()
                total_accuracy += accuracy.item()
    
        avg_val_loss = val_loss / len(test_loader)
        avg_val_accuracy = total_accuracy / len(test_loader) * 100

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.2f}%')


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    #dataset_folder = os.path.join(os.getcwd(), "data")
    #train_dataset = CIFAR10(dataset_folder, train=True, download=True, transform=ToTensor())
    #test_dataset = CIFAR10(dataset_folder, train=False, download=True, transform=ToTensor())
    
    print("Loading extended MNIST dataset...")
    data = torch.load('inverted_mnist_dataset.pt')


    train_dataset = ExtendedMNISTDataset(data['train_images'], data['train_labels'])
    test_dataset = ExtendedMNISTDataset(data['test_images'], data['test_labels'])


    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128,  shuffle=False)

    model = CifarXModel(lr=0.001, wd=0.0001)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model.lr, weight_decay=model.wd)
    
    train_model(model, train_loader, test_loader, optimizer, criterion, epochs=30, device=device)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "checkpoint.pth")


if __name__ == "__main__":
    print("[EXPERIMENT]")
    main()
    # Epoch [10/10], Train Loss: 0.0187, Val Loss: 0.0316, Val Accuracy: 99.02%