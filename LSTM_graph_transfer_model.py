
from graphtransformer.layers.graph_transformer_edge_layer import GraphTransformerLayer
from graphtransformer.layers.mlp_readout_layer import MLPReadout
import dgl
from layers import *
import numpy as np
from sklearn.cluster import SpectralClustering
import math
from torchvision.ops.boxes import _box_inter_union

class GCNnet_JRDB_simin(nn.Module):

    def __init__(self, cfg):
        super(GCNnet_JRDB_simin, self).__init__()
        hidden_dim = 32
        num_heads = 2
        out_dim = 32
        in_feat_dropout = 0.3
        dropout = 0.3
        n_layers = 2
        self.readout = "mean"
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.edge_feat = True
        self.device = 'cuda'
        self.lap_pos_enc = None
        self.wl_pos_enc = None
        if self.lap_pos_enc:
            pos_enc_dim = 8
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        self.embedding_e = nn.Embedding(11, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem
        ####===============================
        self.cfg = cfg
        self.dim = 32
        self.relu = nn.ReLU()
        self.backbone1 =nn.LSTM(input_size=4, hidden_size=32, num_layers=1)
        self.fc_emb_ind = nn.Linear(832 * 25, self.dim)
        self.criterion = nn.MSELoss(reduction='mean')
        self.fc_other = nn.Linear(2 * self.dim, 1)
        # self.attention = SelfAttention(n_channels=832)
        self.att_embed = nn.Linear(1, 16)
        self.num_linear = nn.Linear(32, 16)
        self.linear = nn.Linear(2, 1)
        self.cardinality_1 = nn.Linear(1, 16)
        # self.cardinality_2 = nn.Linear(272, 1)
        self.cardinality_2 = nn.Linear(32, 1)
        self.grouping = nn.Linear(16, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def giou_loss(self,input_boxes, target_boxes, eps=1e-7):
        """
        Args:
            input_boxes: Tensor of shape (N, 4) or (4,).
            target_boxes: Tensor of shape (N, 4) or (4,).
            eps (float): small number to prevent division by zero
        """
        inter, union = _box_inter_union(input_boxes, target_boxes)
        iou = inter / union

        # area of the smallest enclosing box
        min_box = torch.min(input_boxes, target_boxes)
        max_box = torch.max(input_boxes, target_boxes)
        area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])

        giou = iou - ((area_c - union) / (area_c + eps))

        loss = (1 - giou)/ 2.0

        return loss.sum()

    def forward(self, batch_data):
        if batch_data[-1] in ['train', 'val']:
            list_info_tracks, _, _, cluster_ID, _, is_training = batch_data
        else:
            list_info_tracks,_, boxes, bboxes_num, is_training = batch_data


        if  torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        hidden_size=32
        OW=1
        num_track =len(list_info_tracks[-1])
        ###### -----LSTM_------------------------
        track_persons_lstm_Mat = []
        for inx, PLF in enumerate(list_info_tracks[len(list_info_tracks)-1 ]):
            track_person_lstm = []
            for frame in range(len(list_info_tracks)):
                if PLF in list_info_tracks[frame]:
                    T = list_info_tracks[frame][PLF]
                    track_person_lstm.append(T)

            track_person_lstm = torch.Tensor(track_person_lstm).to(device)
            track_person_lstm =torch.unsqueeze(track_person_lstm, 0)
            sequence_length=track_person_lstm.shape[1]
            h0 = torch.randn(1, sequence_length, hidden_size)
            c0 = torch.randn(1, sequence_length, hidden_size)
            output, _ = self.backbone1(track_person_lstm.to(device), (h0.to(device), c0.to(device)))
            feature_lstm_per = output[:, -1]
            if inx == 0:
                track_persons_lstm_Mat = feature_lstm_per
            else:
                track_persons_lstm_Mat = torch.cat((track_persons_lstm_Mat, feature_lstm_per), 0)

        track_persons_lstm_Mat = track_persons_lstm_Mat.to(device="cpu").detach().numpy()
        #### GIOU_Simin_Track---------------------------------------------------
        dic_persons2 = {}
        for inx, PLF in enumerate(list_info_tracks[len(list_info_tracks) - 1]):
            start2 = -1
            end2 = -1
            flag2 = 0
            for frame2 in range(len(list_info_tracks) - 1,-1,-1):
                if PLF in list_info_tracks[frame2] and flag2 == 0:
                    end2 = frame2
                    flag2 = 1
                elif PLF in list_info_tracks[frame2]:
                    start2 = frame2
                elif PLF not in list_info_tracks[frame2]:
                    break
            dic_persons2[PLF] = [start2, end2]
        #### GIOU_Simin_Track------GIOU Track Calculation------------------------------------

        num_tracks_key_frame = len(list_info_tracks[len(list_info_tracks) - 1])
        Matix_GOIU = torch.ones(num_tracks_key_frame, num_tracks_key_frame) * -1

        for inx_i, PLF1 in enumerate(list_info_tracks[len(list_info_tracks)-1]):
            for inx_j, PLF2 in enumerate(list_info_tracks[len(list_info_tracks)-1]):
                if Matix_GOIU[inx_i][inx_j]==-1:
                    max_end=max(dic_persons2[PLF1][1], dic_persons2[PLF2][1])
                    min_start = min(dic_persons2[PLF1][0], dic_persons2[PLF2][0])
                    GIOU=[]
                    for frame1 in range(min_start,max_end+1):
                        if (PLF1 in list_info_tracks[frame1]) and (PLF2 in list_info_tracks[frame1]) :
                            bbx_PLF1=list_info_tracks[frame1][PLF1]
                            bbx_PLF2 = list_info_tracks[frame1][PLF2]

                            bbx_PLF1 = torch.as_tensor(bbx_PLF1)
                            bbx_PLF2 = torch.as_tensor(bbx_PLF2)

                            bbx_PLF1=torch.unsqueeze(bbx_PLF1, 0)
                            bbx_PLF2=torch.unsqueeze(bbx_PLF2, 0)

                            ####------ stittched images's problem
                            if (bbx_PLF1[0][0].item() < bbx_PLF2[0][0].item()) and (
                                    bbx_PLF1[0][2].item() < bbx_PLF2[0][2].item()):
                                bbx_Main = bbx_PLF2.clone()
                                bbx_Slave = bbx_PLF1.clone()
                            else:
                                bbx_Main = bbx_PLF1.clone()
                                bbx_Slave = bbx_PLF2.clone()

                            ####------ stittched images's problem
                            next_bbx_Slave = bbx_Slave.clone()
                            two_next_bbx_Slave = bbx_Slave.clone()

                            next_bbx_Slave[0][0] = bbx_Slave[0][0].item() + OW
                            next_bbx_Slave[0][2] = bbx_Slave[0][2].item() + OW

                            two_next_bbx_Slave[0][0] = bbx_Slave[0][0].item() + (2 * OW)
                            two_next_bbx_Slave[0][2] = bbx_Slave[0][2].item() + (2 * OW)

                            curr_GIOU = min(self.giou_loss(bbx_Main, bbx_Slave),
                                            self.giou_loss(bbx_Main, next_bbx_Slave),
                                            self.giou_loss(bbx_Main, two_next_bbx_Slave))


                        ###----------------------------------------------------
                        else:
                            curr_GIOU=1

                        GIOU.append(curr_GIOU)
                    Mean=sum(GIOU)/((max_end-min_start)+1)
                    if math.isnan(Mean):
                       Matix_GOIU[inx_i,inx_j]=1
                       Matix_GOIU[inx_j,inx_i] = 1
                    else:
                       Matix_GOIU[inx_i, inx_j] = Mean
                       Matix_GOIU[inx_j,inx_i] = Mean


        g = dgl.DGLGraph()
        g.add_nodes(num_track)  # isolated nodes are added
        edge_list = []
        for edge_i in range(num_track):
            for edge_j in range(num_track):
                g.add_edges(edge_i, edge_j)
                edge_list.append(Matix_GOIU[edge_i][edge_j])

        g.ndata['track_feat'] = torch.from_numpy(track_persons_lstm_Mat)
        g.edata['GIOU_feat'] = torch.from_numpy(np.array(edge_list))
        g.to(device)
        h = g.ndata['track_feat']
        e = g.edata['GIOU_feat']
        h = self.in_feat_dropout(h)
        e1 = (torch.tensor(e) * 10).type(torch.LongTensor)
        e2 = self.embedding_e(e1.to(device=device))
        # convnets
        for conv in self.layers:
            h, e2 = conv(g.to(device), h.to(device), e2.to(device))

        H1 = 2*(F.sigmoid(h))-1   # H1 is features nodes

        ###### ------------------------- Residual connection------------------------------------
        output = H1.detach().cpu().numpy() + track_persons_lstm_Mat
        output_lstm = output.squeeze()

        output_feature, _ = torch.max(torch.from_numpy(output_lstm), dim=0)
        output_feature_lstm = self.relu(self.num_linear(output_feature.reshape(1, -1).to(device=device)))


        ######---------------calculating Euclidean distance between node features-------------
        Matix_Euc = torch.ones(track_persons_lstm_Mat.shape[0], track_persons_lstm_Mat.shape[0]) * -1
        for per1 in range(track_persons_lstm_Mat.shape[0]):
            for per2 in range(per1+1, track_persons_lstm_Mat.shape[0]):
                t=np.linalg.norm(H1[per1, :].detach().cpu() - H1[per2, :].detach().cpu())
                Matix_Euc[per1][per2] = torch.tensor(t)
                Matix_Euc[per2][per1] = torch.tensor(t)
        #####______Normilizing Euclidean distance _______________________________________
        Matix_Euc1 = Matix_Euc.cpu().data.numpy()
        Matix_Euc_norm = (Matix_Euc1 - Matix_Euc1.min()) / (Matix_Euc1.max() - Matix_Euc1.min())

        ############ MLP layers
        Matix_GOIU = Matix_GOIU.reshape(-1, 1).to(device=device)
        Matix_Euc_norm = torch.Tensor(Matix_Euc_norm).reshape(-1, 1).to(device=device)
        new_att = self.linear(torch.cat((Matix_GOIU, Matix_Euc_norm), dim=1))
        giou_feature = self.relu(self.cardinality_1(Matix_GOIU))  # [n^2, 16]
        # giou_feature = new_att.clone()
        new_att = torch.sigmoid(new_att)
        giou_feature = torch.sum(giou_feature, dim=0).reshape(-1, 16)  # [1,16]
        cardinality = self.cardinality_2(torch.cat((output_feature_lstm, giou_feature), dim=1))  # [1,1] # [1,

        membership_prediction = []

        if is_training == 'train':
            print("")  # construct train gp_features based on GT connections!
        else:  # if valid or test be

            if int(cardinality.item()) <= 0:
                final_num_pred = 1
            elif int(cardinality.item()) >= num_track:
                final_num_pred = num_track
            else:
                final_num_pred = int(cardinality.item())

            adj_mat = new_att.reshape(int(np.sqrt(new_att.shape[0])),
                                      int(np.sqrt(new_att.shape[0])))
            if adj_mat.shape[0] > 1:

                sc = SpectralClustering(final_num_pred, affinity='precomputed', n_init=100)
                sc.fit(adj_mat.cpu())
                labels = sc.labels_  # Exactly like gt memberships!

            else:

                labels = [0]

            membership_prediction.append(torch.from_numpy(labels).to(device))

        other_scores=[]

        if is_training == 'train':
            return cardinality, new_att, other_scores
        else:
            return cardinality, new_att, membership_prediction, other_scores

