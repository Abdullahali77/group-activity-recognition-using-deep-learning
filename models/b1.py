import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Modify the final fully connected layer to match the number of classes
num_classes = 8
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 8)

# Freeze Layers (layer1)
for layer in [model.layer1]:
    for param in layer.parameters():
        param.requires_grad = False


# Move the model to the GPU 
device = torch.device("cuda")
model.to(device)

# Calculate class weights as inverse frequency
label_counts = Counter([label.item() for _, label in train_dataset])  
total_samples = len(train_dataset)
# Calculate normalized weights for the 8 classes
class_weights = [total_samples / label_counts[i] if i in label_counts else 0 for i in range(num_classes)]

# Convert weights to a tensor and move to the appropriate device
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Define the Loss Function with Weights
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),  lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
