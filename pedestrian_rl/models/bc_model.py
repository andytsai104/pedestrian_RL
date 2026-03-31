import torch
import torch.nn as nn


class BehaviorCloningPloicy(nn.Module):
    '''
    LayerNorm(): ensure no data loss during different scales
    SiLU(): ensuure smooth actions output with all gradient
    Softplus(): ensure speed output is always > 0
    '''
    def __init__(self, cnn_encoder, 
                 bev_feature_dim=128, 
                 scaler_feature_dim=8, 
                 hidden_dim=256, 
                 direction_dim=3):
        
        super().__init__()

        self.cnn_encoder = cnn_encoder
        self.trunk = nn.Sequential(
            nn.Linear(bev_feature_dim + scaler_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, direction_dim)
        )


    def forward(self, data):
        '''
        1. Get data
        2. Concatenate different data
        3. Send to trunk and different head
        4. return policy
        '''
        # Get intended data
        bev_data = data["bev_data"]
        goal_rel = data["goal_rel"]
        velocity = data["velocity"]
        motion_heading = data["motion_heading"].unsqueeze(-1)
        speed = data["speed"].unsqueeze(-1)

        # Get BEV features and states features
        bev_feature = self.cnn_encoder(bev_data)
        scalar_data = torch.cat([
            velocity, 
            speed,
            motion_heading,
            goal_rel
        ], dim=-1)

        fused_data = torch.cat([bev_feature, scalar_data], dim=-1)

        # Get latent features
        latent_features = self.trunk(fused_data)

        # Get predictions
        pred_speed = self.speed_head(latent_features)
        pred_direction = self.direction_head(latent_features)

        return {
            "pred_speed": pred_speed,
            "pred_direction": pred_direction
        }