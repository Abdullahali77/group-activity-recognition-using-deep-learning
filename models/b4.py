import torch
import torch.nn as nn
import torchvision.models as models

class B4_Imp1_Model(nn.Module):
    def __init__(self, b1_weights_path, hidden_dim=256, num_classes=8):
        super(B4_Imp1_Model, self).__init__()
        
        # 1. Setup Base ResNet
        base = models.resnet50()
        
        base.fc = nn.Linear(base.fc.in_features, 8) 
        
        # 2. Load Weights
        if os.path.exists(b1_weights_path):
            try:
                state_dict = torch.load(b1_weights_path, map_location=device)
                base.load_state_dict(state_dict)
                print("Successfully loaded B1 weights!")
            except RuntimeError as e:
                print(f"Weight loading failed: {e}")
                print("Re-initializing with correct shape...")
                # Fallback logic if you are unsure
                base.fc = nn.Linear(base.fc.in_features, 9)
                base.load_state_dict(torch.load(b1_weights_path, map_location=device))
        else:
            print("Warning: B1 weights path not found. Using ImageNet.")

        # 3. Create Feature Extractor (Remove FC)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        
        # Freeze CNN
        for p in self.cnn.parameters():
            p.requires_grad = False
            
        # 4. LSTM
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 5. Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),       
            nn.BatchNorm1d(256),         
            nn.ReLU(),                      
            nn.Dropout(0.5),                 
            nn.Linear(256, num_classes)       
        )
        
    def forward(self, x):
        # Input x: [Batch, Seq_Len, 3, 224, 224]
        b, s, c, h, w = x.size()
        
        # Flatten for CNN: [Batch*Seq_Len, 3, 224, 224]
        x_flat = x.view(b*s, c, h, w)
        
        # Extract Features
        with torch.no_grad(): # Ensure CNN is frozen
            self.cnn.eval()
            cnn_feats = self.cnn(x_flat) # [B*S, 2048, 1, 1]
            
        # Reshape for LSTM: [Batch, Seq_Len, 2048]
        cnn_feats = cnn_feats.view(b, s, 2048)
        
        # LSTM Pass
        # out: [Batch, Seq_Len, Hidden], (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(cnn_feats)
        
        # Take the last hidden state
        final_state = h_n[-1] # [Batch, Hidden]
        
        # Classify
        out = self.fc(final_state)
        return out
