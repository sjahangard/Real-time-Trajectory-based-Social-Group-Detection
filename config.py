import time
import os
import sys

class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Global
        self.image_size = 720, 1280
        self.batch_size = 1
        self.test_batch_size = 1
        self.num_boxes = 50
        
        # Gpu
        self.use_gpu = True
        self.use_multi_gpu = False
        self.device_list = "0"
        
        # Dataset
        assert(dataset_name in ['volleyball', 'collective', 'JRDB'])
        self.dataset_name = dataset_name

        if dataset_name == 'JRDB':
            self.train_annot_path = '/media/fitadmin/HDD/Simin/JRDB_Dataset/JRDB2019/Train/train_labels/labels/labels_2d_stitched'
            self.test_annot_path = '/media/fitadmin/HDD/Simin/JRDB_Dataset/JRDB2019/Test/test_images/images/image_stitched'

            self.train_det_annot_path = '/media/fitadmin/HDD/Simin/JRDB_Dataset/JRDB2019/Train/train_detections/detections/detections_2d_stitched'
            # self.test_det_annot_path = '/home/mahsa/Downloads/detections_2d_stitched'
            # self.test_det_annot_path = '/home/mahsa/Downloads/SOTA/2dd/2021-06-09 04_31_55_submit_dh2'  # better accuracy!
            # self.test_det_annot_path = '/home/mahsa/Downloads/SOTA/3dd/2021-05-28 07_44_37_testing_submission_20.6.12'  # cong
            # self.test_det_annot_path = '/home/mahsa/Downloads/JRMOT_ROS-master/best_det'  # best 3d
            self.test_det_annot_path='/home/mahsa/Downloads/JRMOT_ROS-master/test_mix_4'

            self.train_data_path = '/media/fitadmin/HDD/Simin/JRDB_Dataset/JRDB2019/Train/train_images/images/image_stitched'
            self.test_data_path = '/home/mahsa/Downloads/cvgl/cvgl_with_final_annot_untrimmed/group/jrdb/data/dataset/images/image_stitched'


            self.train_seqs = ['bytes-cafe-2019-02-07_0', 'clark-center-2019-02-28_0',
                               'clark-center-intersection-2019-02-28_0',
                               'cubberly-auditorium-2019-04-22_0', 'forbes-cafe-2019-01-22_0',
                               'gates-159-group-meeting-2019-04-03_0',
                               'gates-basement-elevators-2019-01-17_1', 'gates-to-clark-2019-02-28_1',
                               'hewlett-packard-intersection-2019-01-24_0',
                               'huang-basement-2019-01-25_0', 'huang-lane-2019-02-12_0', 'jordan-hall-2019-04-22_0',
                               'memorial-court-2019-03-16_0',
                               'packard-poster-session-2019-03-20_0', 'packard-poster-session-2019-03-20_1',
                               'packard-poster-session-2019-03-20_2',
                               'stlc-111-2019-04-19_0', 'svl-meeting-gates-2-2019-04-08_0',
                               'svl-meeting-gates-2-2019-04-08_1', 'tressider-2019-03-16_0']

            self.valid_seqs = ['clark-center-2019-02-28_1', 'gates-ai-lab-2019-02-08_0', 'huang-2-2019-01-25_0',
                               'meyer-green-2019-03-16_0', 'nvidia-aud-2019-04-18_0',
                               'tressider-2019-03-16_1', 'tressider-2019-04-26_2']

            self.test_seqs = ['cubberly-auditorium-2019-04-22_1', 'discovery-walk-2019-02-28_0', 'discovery-walk-2019-02-28_1', 'food-trucks-2019-02-12_0',
                              'gates-ai-lab-2019-04-17_0', 'gates-basement-elevators-2019-01-17_0', 'gates-foyer-2019-01-17_0', 'gates-to-clark-2019-02-28_0',
                              'hewlett-class-2019-01-23_0', 'hewlett-class-2019-01-23_1', 'huang-2-2019-01-25_1', 'huang-intersection-2019-01-22_0',
                              'indoor-coupa-cafe-2019-02-06_0', 'lomita-serra-intersection-2019-01-30_0', 'meyer-green-2019-03-16_1', 'nvidia-aud-2019-01-25_0',
                              'nvidia-aud-2019-04-18_1', 'nvidia-aud-2019-04-18_2', 'outdoor-coupa-cafe-2019-02-06_0', 'quarry-road-2019-02-28_0',
                              'serra-street-2019-01-30_0', 'stlc-111-2019-04-19_1', 'stlc-111-2019-04-19_2', 'tressider-2019-03-16_2', 'tressider-2019-04-26_0',
                              'tressider-2019-04-26_1', 'tressider-2019-04-26_3']

            # self.train_seqs = ['bytes-cafe-2019-02-07_0']
            # self.valid_seqs = ['bytes-cafe-2019-02-07_0']
            # self.test_seqs = ['bytes-cafe-2019-02-07_0']
        # Backbone 
        self.backbone = 'i3d'
        self.crop_size = 5, 5
        self.train_backbone = False
        self.out_size = 30, 45
        self.emb_features = 832

        # Activity Action
        self.num_actions = 9
        self.num_activities = 8
        self.actions_loss_weight = 1.0
        self.actions_weights = None

        # Sample
        self.num_frames = 8
        self.num_before = 8
        self.num_after = 0

        # GCN
        self.num_features_boxes = 1024
        self.num_features_relation = 256
        self.num_graph = 1
        self.num_features_gcn = self.num_features_boxes
        self.gcn_layers = 1
        self.tau_sqrt = False
        self.pos_threshold = 0.2  #distance mask threshold in position relation

        # Training Parameters
        self.train_random_seed = 42
        self.train_learning_rate = 2e-4
        self.lr_plan = {41:1e-4, 81:5e-5, 121:1e-5}
        self.train_dropout_prob = 0.3
        self.weight_decay = 0  #l2 weight decay
    
        self.max_epoch = 150
        self.test_interval_epoch = 2
        
        # Exp
        self.training_stage = 1
        self.stage1_model_path = ''
        self.test_before_train = False
        self.exp_note = 'Group-Activity-Recognition'
        self.exp_name = None
        
        
    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '[%s_stage%d][%s]'%(self.exp_note, self.training_stage, time_str)
            
        self.result_path = 'result/%s'%self.exp_name
        self.log_path = 'result/%s/log.txt'%self.exp_name
            
        if need_new_folder:
            os.makedirs(self.result_path)
