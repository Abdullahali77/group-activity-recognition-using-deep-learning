import os
# Silence the Albumentations update warning
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import Hierarchical_Group_Activity_DataSet, collate_fn, activities_labels
from model import Hierarchical_Group_Activity_Classifer

# Configuration
DATA_DIR = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball" 
BATCH_SIZE = 4 # Per GPU (Total = 8)
EPOCHS = 85
LR = 6e-6
WEIGHT_DECAY = 1

train_split = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_split = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_process(rank, world_size):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # Model Setup
    model = Hierarchical_Group_Activity_Classifer().to(device)
    # === NEW RESUME BLOCK ===
    # Update this path to match exactly where you uploaded the file in Step 1!
    RESUME_PATH = "/kaggle/input/models/abdullahali7/b9/pytorch/default/1/best_b9_lstm_model.pth" 
    if os.path.exists(RESUME_PATH):
        if rank == 0:
            print(f"LOADING PREVIOUS BRAINPOWER FROM: {RESUME_PATH}")
        # map_location safely loads it across the multiple GPUs
        model.load_state_dict(torch.load(RESUME_PATH, map_location=device, weights_only=True))
    # ========================
    model = DDP(model, device_ids=[rank])
    
    # Augmentations
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.OneOf([A.GaussianBlur((3, 7)), A.ColorJitter(0.2), A.MotionBlur(5)], p=0.55),
        A.HorizontalFlip(p=0.5), # Standard flip is usually fine for Volleyball
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Datasets & Samplers
    train_dataset = Hierarchical_Group_Activity_DataSet(f"{DATA_DIR}/videos", f"{DATA_DIR}/annot_all.pkl", train_split, activities_labels, train_transforms)
    val_dataset = Hierarchical_Group_Activity_DataSet(f"{DATA_DIR}/videos", f"{DATA_DIR}/annot_all.pkl", val_split, activities_labels, val_transforms)
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Loss & Optimizer
    person_criterion = nn.CrossEntropyLoss(label_smoothing=0.10).to(device)
    group_criterion = nn.CrossEntropyLoss().to(device) 
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Updated AMP Syntax
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15) 

    best_acc = 0.0

    # Training Loop
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, person_labels, group_labels) in enumerate(train_loader):
            inputs, person_labels, group_labels = inputs.to(device), person_labels.to(device), group_labels.to(device)
            optimizer.zero_grad()
            
            # Updated AMP Syntax
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss_1 = person_criterion(outputs['person_output'], person_labels)
                loss_2 = group_criterion(outputs['group_output'], group_labels)
                
                # The 0.60 Dual-Loss Strategy
                loss = loss_2 + (0.60 * loss_1)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # --- THE HEARTBEAT PRINT ---
            if rank == 0 and batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Current Batch Loss: {loss.item():.4f}")
            # ---------------------------
            
        # Validation
        model.eval()
        correct, total = 0, 0
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Training done. Running Validation...")
            
        with torch.no_grad():
            for inputs, person_labels, group_labels in val_loader:
                inputs, group_labels = inputs.to(device), group_labels.to(device)
                outputs = model(inputs)
                
                _, predicted = outputs['group_output'].max(1)
                _, target_class = group_labels.max(1)
                
                total += group_labels.size(0)
                correct += predicted.eq(target_class).sum().item()
        
        # Aggregate metrics across GPUs
        correct_tensor = torch.tensor(correct).to(device)
        total_tensor = torch.tensor(total).to(device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        val_acc = 100. * correct_tensor.item() / total_tensor.item()
        scheduler.step(val_acc)
        
        if rank == 0:
            print(f"====> Epoch [{epoch+1}/{EPOCHS}] COMPLETE | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']}")
            if val_acc > best_acc:
                best_acc = val_acc
                
                # --- NEW FOLDER CREATION LINE ---
                os.makedirs("/kaggle/working/outputs", exist_ok=True) 
                
                torch.save(model.module.state_dict(), "/kaggle/working/outputs/best_b9_lstm_model.pth")
                print("--> Saved Best Model!")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Warning: Only 1 GPU detected. DDP requires 2 for T4x2 setup.")
    mp.spawn(train_process, args=(world_size,), nprocs=world_size, join=True)
