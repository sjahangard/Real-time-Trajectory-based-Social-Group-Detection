import torch.optim as optim
from dataset import *
from LSTM_graph_transfer_model import *
from base_model import *
from utils import *
from tensorboardX import SummaryWriter
from jrdb_toolkit.Action_Social_grouping_eval.JRDB_eval import *
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
def disable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()
def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
def train_net_simin(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # Reading dataset
    training_set, validation_set = return_dataset(cfg)
    # __________________________________________________________________
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 0
    }
    training_loader = data.DataLoader(training_set, drop_last=True, **params)  # len = 1746
    params['batch_size'] = cfg.test_batch_size
    params['shuffle'] = False
    name_output=cfg.name_output
    validation_loader = data.DataLoader(validation_set, drop_last=True, **params)
    # ______________________________________________________________________________
    ###Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    np.random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    gcnnet_list = {'JRDB': GCNnet_JRDB_simin}

    if cfg.training_stage == 1:
        GCNnet_simin = gcnnet_list[cfg.dataset_name]
        model = GCNnet_simin(cfg)

    else:
        assert (False)

    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device=device)
    # ___________________________________________________________________________________________
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,
                           weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion_3 = torch.nn.BCELoss()

    # tensorboard
    train_logger1 = SummaryWriter(log_dir=os.path.join('./logs/'+name_output, 'train'))
    train_eval_logger1 = SummaryWriter(log_dir=os.path.join('./logs/'+name_output,'train_eval'))

    # Training iteration
    start_epoch =0
    max_epoch = 50

# _________________________________________________________________________________________________________________
    for epoch in range(start_epoch, max_epoch):
        # One epoch of forward and backward
        print('Epoch {}/{}'.format(epoch,  max_epoch - 1))

        train_info = train_JRDB(training_loader, model, device,criterion_3, optimizer, epoch,name_output)
        train_logger1.add_scalars("train_log", train_info, epoch)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            }
        filepath = cfg.result_path + '/stage%d_epoch_%s%d.pth' % (cfg.training_stage, name_output, epoch)
        torch.save(state, filepath)
        print('model saved to:', filepath)
        print("**********************Epoch***********************",epoch)

        if epoch%5==0:

            val_info = val_JRDB(validation_loader, model, device, criterion_3, epoch,name_output)
            train_eval_logger1.add_scalars("train_eval", val_info, epoch)
        scheduler.step(train_info['loss'])

def train_JRDB(data_loader, model, device, criterion_3, optimizer, epoch,name_output):

    epoch_timer = Timer()
    loss_meter = AverageMeter()
    mem_loss = AverageMeter()
    eigen_loss = AverageMeter()
    num_loss = AverageMeter()  #cardinality
    is_training = 'train'
    model.train()
    model.apply(set_bn_eval)

    for idx, batch_data in enumerate(data_loader):
        list_info_tracks=batch_data[0]
        images = batch_data[1]
        cluster_ID=batch_data[3]
        print(name_output + ' is running')
        batch_size = 1
        num_track = len(list_info_tracks[len(list_info_tracks) - 1])
        bboxes = batch_data[2]
        bboxes_num = batch_data[4].reshape(batch_size, 1)
        num_tracks_key_frame = len(list_info_tracks[len(list_info_tracks) - 1])

        cardinality, new_att, _ = model((list_info_tracks,images, bboxes, cluster_ID, bboxes_num, is_training))

        # _________________________________________________________ Social _ grouping
        cluster_ID_nopad, refined_predictions_nopad = [], []
        cluster=[]
        for x, per in enumerate(cluster_ID):
            cluster.append(cluster_ID[per])

        matrix = torch.eye(num_tracks_key_frame, num_tracks_key_frame).to(device=device)
        for j in range(num_tracks_key_frame):
            for t in range(j + 1, num_tracks_key_frame):
                if cluster[j] == cluster[t]:
                    matrix[j][t] = 1
                    matrix[t][j] = 1

        cluster_ID_nopad.append(matrix.reshape(-1))
        final_cluster_ID = torch.cat(cluster_ID_nopad, dim=0).reshape(num_tracks_key_frame * num_tracks_key_frame, 1)  # 0 or 1

        sample_loss = 0
        pred_weights = new_att.reshape(num_tracks_key_frame, num_tracks_key_frame)
        degree_matrix = torch.diag(torch.sum(matrix, dim=1))
        laplacian_matrix = degree_matrix - matrix
        eigenvalues, eigenvectors = torch.eig(laplacian_matrix, eigenvectors=True)
        eigenvalues = eigenvalues[:, 0]
        index = []
        for ind, i in enumerate(eigenvalues):
            if i <= 10 ** -5:
                index.append(ind)

        num_clusters = len(index)
        zero_eigenvectors = torch.zeros((num_clusters, num_tracks_key_frame)).to(device=device)
        for ind, i in enumerate(index):
            zero_eigenvectors[ind, :] = eigenvectors[:, i]
        # ________________GT____EIGEN________CALC______________#
        alpha = 1
        beta = 1
        for i in range(zero_eigenvectors.shape[0]):
            e_gt = zero_eigenvectors[i, :].reshape(num_tracks_key_frame, 1)
            e_gt_t = torch.t(e_gt)
            pred_weights_t = torch.t(pred_weights)
            d_term = torch.chain_matmul(e_gt_t, pred_weights_t, pred_weights, e_gt)
            e_hat = torch.eye(num_tracks_key_frame).to(device=device) - torch.matmul(e_gt, e_gt_t)
            e_hat_plus = torch.matmul(pred_weights, e_hat)
            e_hat_plus_t = torch.t(e_hat_plus)
            e_neg = torch.chain_matmul(e_hat_plus_t, e_hat_plus)
            r_term = torch.trace(e_neg)
            sample_loss += (d_term + (alpha * torch.exp(-1 * beta * r_term)))  # Social_Grouping (2)L_eig (L,L')

        GT_num_clusters = torch.Tensor([len(torch.unique(torch.as_tensor(cluster)))]).to(device=device).reshape(1, 1)  # what if -1?!
        cardinality_loss = F.mse_loss(cardinality, GT_num_clusters.detach())  # Social_Grouping (3)L_MSE (h0,GT)
        membership_loss = criterion_3(new_att, final_cluster_ID.detach())  # Social_Grouping (1)L_BCE (A, A')
        sample_loss = sample_loss / zero_eigenvectors.shape[0]  # Social_Grouping (2)L_eig (L,L')

        total_loss = cardinality_loss * 0.0005 + membership_loss * 0.1 + sample_loss * 0.0005
        loss_meter.update(total_loss.item(), batch_size)
        eigen_loss.update(sample_loss.item()*0.0005 , batch_size)
        mem_loss.update(membership_loss.item()*0.1 , batch_size)
        num_loss.update(cardinality_loss.item()* 0.0005, batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_info = {
            'time': epoch_timer.timeit(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'membership_loss': mem_loss.avg,
            'eigen_loss': eigen_loss.avg,
            'num_loss': num_loss.avg,
        }
    print(train_info)
    return train_info
# -----------------------------------------------------------------------------------------------------------------------
def read_labelmap(labelmap_file):
    labelmap = {}
    class_ids = set()
    name = ""
    labelmap_file = open(labelmap_file)
    for line in labelmap_file:
        if line.startswith("  name:"):
            name = line.split('"')[1]
        elif line.startswith("  id:") or line.startswith("  label_id:"):
            class_id = int(line.strip().split(" ")[-1])
            labelmap[name] = class_id
            class_ids.add(class_id)
    return labelmap, class_ids
# _______________________________________________________________________________________________________________________
def val_JRDB(data_loader, model, device,criterion_3, epoch,name_output):

    epoch_timer = Timer()
    loss_meter = AverageMeter()
    mem_loss = AverageMeter()
    eigen_loss = AverageMeter()
    num_loss = AverageMeter()  # cardinality
    is_training = 'val'
    model.eval()

    output = './pred/val/'
    seq = ['clark-center-2019-02-28_1', 'gates-ai-lab-2019-02-08_0', 'huang-2-2019-01-25_0',
           'meyer-green-2019-03-16_0', 'nvidia-aud-2019-04-18_0',
           'tressider-2019-03-16_1', 'tressider-2019-04-26_2']

    seq_ID = {a: i for i, a in enumerate(seq)}
    with torch.no_grad():
        for idx, batch_data in enumerate(data_loader):
            # prepare batch data
            sid = batch_data[-2][0]
            fid = batch_data[-1].item()
            list_info_tracks = batch_data[0]
            images = batch_data[1]
            cluster_ID = batch_data[3]
            batch_size = 1
            num_track = len(list_info_tracks[len(list_info_tracks) - 1])
            bboxes = batch_data[2]
            bboxes_num = batch_data[4]
            # forward
            cardinality, new_att, membership_predictions, _ = model(
                (list_info_tracks,images, bboxes, cluster_ID, bboxes_num, is_training))

            cluster_ID_nopad, refined_predictions_nopad = [], []
            cluster = []
            for x, per in enumerate(cluster_ID):
                cluster.append(cluster_ID[per])

            sample_loss = 0
            pred_weights = new_att.reshape(num_track, num_track)

            matrix = torch.eye(num_track, num_track).to(device=device)
            for j in range(num_track - 1):
                for t in range(j + 1, num_track):
                    if cluster[j] == cluster[t]:
                        matrix[j][t] = 1
                        matrix[t][j] = 1

            cluster_ID_nopad.append(matrix.reshape(-1))
            final_cluster_ID = torch.cat(cluster_ID_nopad, dim=0).reshape(num_track * num_track, 1)  # 0 or 1

            degree_matrix = torch.diag(torch.sum(matrix, dim=1))
            laplacian_matrix = degree_matrix - matrix
            eigenvalues, eigenvectors = torch.eig(laplacian_matrix, eigenvectors=True)
            eigenvalues = eigenvalues[:, 0]
            index = []
            for ind, i in enumerate(eigenvalues):
                if i <= 10 ** -5:
                    index.append(ind)

            num_clusters = len(index)
            zero_eigenvectors = torch.zeros((num_clusters, num_track)).to(device=device)
            for ind, i in enumerate(index):
                zero_eigenvectors[ind, :] = eigenvectors[:, i]
            # ________________GT____EIGEN________CALC______________#
            alpha = 1
            beta = 1
            for i in range(zero_eigenvectors.shape[0]):
                e_gt = zero_eigenvectors[i, :].reshape(num_track, 1)
                e_gt_t = torch.t(e_gt)
                pred_weights_t = torch.t(pred_weights)
                d_term = torch.chain_matmul(e_gt_t, pred_weights_t, pred_weights, e_gt)
                e_hat = torch.eye(num_track).to(device=device) - torch.matmul(e_gt, e_gt_t)
                e_hat_plus = torch.matmul(pred_weights, e_hat)
                e_hat_plus_t = torch.t(e_hat_plus)
                e_neg = torch.chain_matmul(e_hat_plus_t, e_hat_plus)
                r_term = torch.trace(e_neg)
                sample_loss += (d_term + (alpha * torch.exp(-1 * beta * r_term)))

            GT_num_clusters = torch.Tensor([len(torch.unique(torch.as_tensor(cluster)))]).to(device=device).reshape(1,1)  # what if -1?!

            cardinality_loss = F.mse_loss(cardinality, GT_num_clusters.detach())
            membership_loss = criterion_3(new_att, final_cluster_ID.detach())
            sample_loss = sample_loss / zero_eigenvectors.shape[0]
            total_loss = cardinality_loss*0.005  + membership_loss*0.05  + sample_loss*0.05

            loss_meter.update(total_loss.item(), batch_size)
            mem_loss.update(membership_loss.item() *0.05, batch_size)
            eigen_loss.update(sample_loss.item()*0.05 , batch_size)
            num_loss.update(cardinality_loss.item()*0.005 , batch_size)
            dc = 1
            Text_file='mem/%d_det_group_'+name_output+'.txt'
            for inx, PLF in enumerate(list_info_tracks[len(list_info_tracks)-1 ]):
                box=list_info_tracks[len(list_info_tracks)-1][PLF]
                pred_list = [seq_ID[sid], fid, box[0].item(), box[1].item(), box[2].item(), box[3].item(),
                             membership_predictions[0][inx].item(),
                             dc, dc]
                str_to_be_added = [str(k) for k in pred_list]
                str_to_be_added = (" ".join(str_to_be_added))
                f = open(output + Text_file % (int(epoch)), "a+")
                f.write(str_to_be_added + "\r\n")
                f.close()

            # __________________________________________________________________________________________________________
        task = "task_3"
        labelmap = "jrdb_toolkit/Action_Social_grouping_eval/label_map/task_3.pbtxt"
        groundtruth = "pred/val/mem/gt_group.txt"
        labelmap_data = open(labelmap, "r+", encoding='utf-8-sig')
        detections_data = open(output +Text_file % (int(epoch)), "r+",
                               encoding='utf-8-sig')
        groundtruth_data = open(groundtruth, "r+", encoding='utf-8-sig')
        metrics = evaluate(labelmap_data, groundtruth_data, detections_data, task)

        mAP1_all1=metrics['all']['PascalBoxes_Precision/mAP@0.5IOU']
        mAP1_G11=metrics['all']['PascalBoxes_PerformanceByCategory/AP@0.5IOU/1']
        mAP1_G21=metrics['all']['PascalBoxes_PerformanceByCategory/AP@0.5IOU/2']
        mAP1_G31=metrics['all']['PascalBoxes_PerformanceByCategory/AP@0.5IOU/3']
        mAP1_G41=metrics['all']['PascalBoxes_PerformanceByCategory/AP@0.5IOU/4']
        mAP1_G51=metrics['all']['PascalBoxes_PerformanceByCategory/AP@0.5IOU/5']


        task2 = "task_2"
        labelmap2 = "jrdb_toolkit/Action_Social_grouping_eval/label_map/task_2.pbtxt"
        labelmap_data2 = open(labelmap2, "r+", encoding='utf-8-sig')
        metrics2 = evaluate(labelmap_data2, groundtruth_data, detections_data, task2)
        mAP21=metrics2['all']['PascalBoxes_Precision/mAP@0.5IOU']
        val_info = {
            'time': epoch_timer.timeit(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'mAP1_G1' : mAP1_G11,
            'mAP1_G2' : mAP1_G21,
            'mAP1_G3' : mAP1_G31,
            'mAP1_G4' : mAP1_G41,
            'mAP1_G5' : mAP1_G51,
            'mAP1_all' : mAP1_all1,
            'mAP2': mAP21,
            'membership_loss': mem_loss.avg,
            'num_loss': num_loss.avg,
            'eigen_loss': eigen_loss.avg,
        }
    print(val_info)
    return val_info

