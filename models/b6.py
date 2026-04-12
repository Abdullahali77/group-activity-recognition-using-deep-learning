import torch
import torch.nn as nn
class Baseline6(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_classes=8):
        super(Baseline6, self).__init__()
        
        # LSTM for frame-level temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True 
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x_pooled, _ = torch.max(x, dim=2) 
        self.lstm.flatten_parameters()
        
        # LSTM output is [Batch, Time, Hidden*2]
        lstm_out, _ = self.lstm(x_pooled)
        
        # Max Pool over TIME as well (Global Temporal Pooling)
        # Instead of just taking the last step, we take the strongest signal from the whole clip
        final_rep, _ = torch.max(lstm_out, dim=1) 
        
        logits = self.classifier(final_rep)
        return logits