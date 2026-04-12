import torch
import torch.nn as nn
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BaselineB3_Fixed(nn.Module):
    def __init__(self, person_model_path, num_classes=8):
        super(BaselineB3_Fixed, self).__init__()
        
        # Load Phase 1 Weights
        base = models.resnet50()
        base.fc = nn.Linear(base.fc.in_features, 9)
        base.load_state_dict(torch.load(person_model_path, map_location=device))
        
        # Frozen Backbone
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        for p in self.backbone.parameters(): p.requires_grad = False
            
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        self.backbone.eval()
        
        b, p, c, h, w = x.size()
        x = x.view(b*p, c, h, w)
        
        with torch.no_grad():
            feats = self.backbone(x)
            
        feats = feats.view(b, p, 2048)
        scene_feat, _ = torch.max(feats, dim=1) # Max Pooling
        out = self.classifier(scene_feat)
        return out