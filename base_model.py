import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import matplotlib.pyplot as plt
from backbone import *
from utils import *
#from roi_align.roi_align import RoIAlign      # RoIAlign module
#from roi_align.roi_align import CropAndResize # crop_and_resize module
import cv2

#_____________________________________________________________________________________________________________________________________________________________________

class SelfAttention(nn.Module):
    "Self attention layer for nd."
    def __init__(self, n_channels:int):
        super(SelfAttention, self).__init__()

        def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
            "Create and initialize a `nn.Conv1d` layer with spectral normalization."
            conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
            nn.init.kaiming_normal_(conv.weight)
            if bias: conv.bias.data.zero_()
            return nn.utils.spectral_norm(conv)

        self.query = conv1d(n_channels, n_channels//8)
        self.key   = conv1d(n_channels, n_channels//8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
#_____________________________________________________________________________________________________________________________________________________________________

class Basenet_collective(nn.Module):
    """
    main module of base model for collective dataset
    """
    def __init__(self, cfg, InceptionI3d):
        super(Basenet_collective, self).__init__()
        self.cfg = cfg
        
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        self.backbone = InceptionI3d(400, in_channels=3)
        self.backbone.load_state_dict(torch.load('./data/pretrained_model/rgb_imagenet.pt'))
        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_action = nn.Linear(K*K*D, NFB)
        self.fc_emb_activity = nn.Linear(61200, NFB) #122400
        self.dropout_emb = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFB, self.cfg.num_activities)

        self.attention = SelfAttention(n_channels=832)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self,filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'attention_state_dict': self.attention.state_dict(),
            'fc_emb_action_state_dict': self.fc_emb_action.state_dict(),
            'fc_emb_activity_state_dict': self.fc_emb_activity.state_dict(),
            'fc_actions_state_dict': self.fc_actions.state_dict(),
            'fc_activities_state_dict': self.fc_activities.state_dict()
        }
        
        torch.save(state, filepath)
        print('model saved to:',filepath)

    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.attention.load_state_dict(state['attention_state_dict'])
        self.fc_emb_action.load_state_dict(state['fc_emb_action_state_dict'])
        self.fc_emb_activity.load_state_dict(state['fc_emb_activity_state_dict'])
        self.fc_actions.load_state_dict(state['fc_actions_state_dict'])
        self.fc_activities.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ',filepath)

    def forward(self, batch_data):

        images_in, boxes_in, bboxes_num_in = batch_data #[16, 17, 3, 480, 720] #[16, 17, 13, 4] #[16, 17]

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes #13
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        EPS = 1e-5
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        
        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B*T, 3, H, W))
        images_in_flat = prep_images(images_in_flat)  # noremalization [-1,1]
        images_in_flat = torch.reshape(images_in_flat, (B, T, 3, H, W))
        images_in_flat = images_in_flat.permute(0, 2, 1, 3, 4).float()

        global_output, local_output = self.backbone(images_in_flat) #[1, 400, 2, 9, 17] # [1, 832, 5, 30, 45]
        boxes_in = boxes_in[:, 8, :, :]
        boxes_in_flat = torch.reshape(boxes_in, (B*MAX_N, 4))  #B*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B*MAX_N,))  #B*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False

        #local_output = local_output.reshape(B, -1, 30, 45).contiguous()

        local_output = local_output[:, :, 2, :, :].contiguous()
        boxes_features_all = self.roi_align(local_output, boxes_in_flat, boxes_idx_flat)  #B*MAX_N, D, K, K, #[13, 4160, 5, 5]
        boxes_features_all = self.dropout_emb(self.attention(boxes_features_all))
        boxes_features_all = boxes_features_all.reshape(B, MAX_N, -1)  #B,MAX_N, D*K*K #[bs, 13, -1]

        # Embedding 
        boxes_features_all = self.fc_emb_action(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all = F.relu(boxes_features_all)
        boxes_features_all = self.dropout_emb(boxes_features_all) #[bs, 13, 1024]
    
        actions_scores = []
        bboxes_num_in = bboxes_num_in[:, 8].reshape(B,)  #B,

        for bt in range(B):
            N = bboxes_num_in[bt]
            boxes_features = boxes_features_all[bt, :N, :].reshape(1, N, NFB)  #1,N,NFB
            boxes_states = boxes_features
            NFS = NFB
            # Predict actions
            boxes_states_flat = boxes_states.reshape(-1, NFS)  #1*N, NFS
            actn_score = self.fc_actions(boxes_states_flat)  #1*N, actn_num
            actions_scores.append(actn_score)

        actions_scores = torch.cat(actions_scores, dim=0)  #ALL_N,actn_num

        # Predict activities
        global_output = global_output.reshape(B, -1)  # B*T*N, D*K*K
        global_output = self.fc_emb_activity(global_output)  # B*T*N, NFB
        global_output = F.relu(global_output)
        global_output = self.dropout_emb(global_output)
        activities_scores = self.fc_activities(global_output)

        return actions_scores, activities_scores
