import torch
import torch.nn as nn


class BehaviorCloningPolicy(nn.Module):
    '''
    BC policy using BEV features + local-frame scalar features.

    Scalar inputs:
        velocity_local   : (B, 2)  -> [right, forward]
        speed            : (B, 1)
        yaw_sin          : (B, 1)
        yaw_cos          : (B, 1)
        goal_rel_local   : (B, 2)  -> [right, forward]
    
    Model Structures:
        LayerNorm(): ensure no data loss during different scales
        SiLU(): ensuure smooth actions output with all gradient
        Softplus(): ensure speed output is always > 0
    '''

    def __init__(
        self,
        cnn_encoder,
        bev_feature_dim=128,
        scalar_feature_dim=7,
        hidden_dim=256,
        direction_dim=2,
        dropout=0.10,
    ):
        super().__init__()

        self.cnn_encoder = cnn_encoder

        self.trunk = nn.Sequential(
            nn.Linear(bev_feature_dim + scalar_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, direction_dim),
        )

    def forward(self, data):
        '''
        1. Get data
        2. Concatenate different data
        3. Send to trunk and different head
        4. return policy
        '''
        bev_data = data["bev_data"]
        velocity_local = data["velocity_local"]
        speed = data["speed"].unsqueeze(-1)
        yaw_sin = data["yaw_sin"].unsqueeze(-1)
        yaw_cos = data["yaw_cos"].unsqueeze(-1)
        goal_rel_local = data["goal_rel_local"]

        bev_feature = self.cnn_encoder(bev_data)

        scalar_feature = torch.cat(
            [velocity_local, speed, yaw_sin, yaw_cos, goal_rel_local],
            dim=-1,
        )

        fused_feature = torch.cat([bev_feature, scalar_feature], dim=-1)
        latent_feature = self.trunk(fused_feature)

        pred_speed = self.speed_head(latent_feature)
        pred_direction = self.direction_head(latent_feature)

        return {
            "pred_speed": pred_speed,
            "pred_direction": pred_direction,
        }
