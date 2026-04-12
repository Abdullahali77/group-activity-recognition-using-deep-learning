import torch
import torch.nn as nn
class BaselineB7(nn.Module):
    def __init__(self, input_size=2048, hidden_size_player=512, hidden_size_frame=512, num_classes=8):
        super(BaselineB7, self).__init__()
        
        # 1. Player LSTM
        # Tracks individual player actions over time
        self.player_lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size_player, 
            num_layers=1, 
            batch_first=True
        )
        
        # 2. Pooling Layer (Adaptive Max Pool)
        # We pool over the 'Players' dimension to get 1 vector per frame
        self.pool = nn.AdaptiveMaxPool1d(1) 
        
        # 3. Frame LSTM
        # Tracks the evolution of the whole scene
        self.lstm_frame = nn.LSTM(
            input_size=hidden_size_player, 
            hidden_size=hidden_size_frame, 
            num_layers=1, 
            batch_first=True
        )
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_frame, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Input: [Batch, Time, Players, Features]
        b, t, p, f = x.size()

        # A. Player Level Analysis
        # Flatten Batch and Players to run LSTM on all players at once
        x = x.view(b * p, t, f) 
        player_output, _ = self.player_lstm(x) # [B*P, T, Hidden_Player]
        
        # Reshape back to separate Batch, Time, and Players
        # [B, P, T, Hidden_Player] -> Permute to [B, T, P, Hidden_Player]
        player_output = player_output.view(b, p, t, -1).permute(0, 2, 1, 3)
        
        # B. Pooling (Spatial Aggregation)
        # Flatten Batch and Time to pool over players for every single frame
        player_output_flat = player_output.reshape(b * t, p, -1)
        
        # Permute for Pooling: [N, Channels, Length] -> [B*T, Hidden_Player, Players]
        player_output_flat = player_output_flat.permute(0, 2, 1)
        
        # Pool -> [B*T, Hidden_Player, 1]
        frame_rep = self.pool(player_output_flat).squeeze(-1) 
        
        # C. Frame Level Analysis
        # Reshape for Frame LSTM: [Batch, Time, Hidden_Player]
        frame_rep = frame_rep.view(b, t, -1)
        
        # Run Frame LSTM
        frame_out, _ = self.lstm_frame(frame_rep) # [Batch, Time, Hidden_Frame]
        
        # Take Last Frame
        final_rep = frame_out[:, -1, :] 
        
        # D. Classify
        logits = self.classifier(final_rep)
        return logits