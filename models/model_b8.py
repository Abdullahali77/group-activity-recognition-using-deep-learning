import torch
import torch.nn as nn
import torchvision.models as models

class Hierarchical_Group_Activity_Classifer(nn.Module):
    def __init__(self, person_num_classes=9, group_num_classes=8, hidden_size=512, num_layers=2):
        super(Hierarchical_Group_Activity_Classifer, self).__init__()

        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            *list(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).children())[:-1]
        )
        
        # Person-Level LSTM
        self.layer_norm_1 = nn.LayerNorm(512)
        self.lstm_1 = nn.LSTM(
            input_size=512, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True, dropout=0.5
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, person_num_classes)
        )

        # Group-Level LSTM
        self.layer_norm_2 = nn.LayerNorm(512)
        self.pool = nn.AdaptiveMaxPool2d((1, 256))
     
        self.lstm_2 = nn.LSTM(
            input_size=512, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True, dropout=0.5
        )
        
        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, group_num_classes)
        )
    
    def forward(self, x):
        b, bb, seq, c, h, w = x.shape  
        x = x.view(b * bb * seq, c, h, w)  
        x1 = self.feature_extractor(x) 

        x1 = x1.view(b * bb, seq, -1)       
        x1 = self.layer_norm_1(x1)          
        
        # LSTM Fix: Unpack the tuple properly
        x2, (h_1, c_1) = self.lstm_1(x1) 

        y1 = self.fc_1(x2[:, -1, :])  

        x = torch.cat([x1, x2], dim=2) 
        x = x.contiguous()             
       
        x = x.view(b * seq, bb, -1) 
        team_1 = x[:, :6, :]      
        team_2 = x[:, 6:, :]      

        team_1 = self.pool(team_1) 
        team_2 = self.pool(team_2) 
        x = torch.cat([team_1, team_2], dim=1)  
       
        x = x.view(b, seq, -1) 
        x = self.layer_norm_2(x) 
        
        # LSTM Fix: Unpack the tuple properly
        x, (h_2, c_2) = self.lstm_2(x) 

        x = x[:, -1, :]     
        y2 = self.fc_2(x)   
        
        return {'person_output': y1, 'group_output': y2}
