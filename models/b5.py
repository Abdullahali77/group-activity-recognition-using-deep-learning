import torch
import torch.nn as nn

class BaselineB5(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_classes=8):
        super(BaselineB5, self).__init__()
        
        # 1. Player LSTM (Shared weights for all players)
        # Learns "What is THIS player doing over time?"
        self.player_lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # 2. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Input x: [Batch, Time, Players, Features]
        b, t, p, f = x.size()
        
        # Flatten to treat every player in the batch independently
        # Shape: [Batch * Players, Time, Features]
        x_flat = x.permute(0, 2, 1, 3).reshape(b * p, t, f)
        
        # Run LSTM per player
        _, (h_n, _) = self.player_lstm(x_flat)
        
        # Take the Last Hidden State (Summary of the player's action)
        # Shape: [Batch * Players, Hidden]
        player_summaries = h_n[-1]
        
        # Reshape to recover batches
        # Shape: [Batch, Players, Hidden]
        player_summaries = player_summaries.view(b, p, -1)
        
        # MAX POOLING over Players (Dim 1)
        # Aggregates individual actions into a scene vector
        # Shape: [Batch, Hidden]
        scene_vector, _ = torch.max(player_summaries, dim=1)
        
        # Classify
        logits = self.classifier(scene_vector)
        return logits