import sys
sys.path.append(".")
from train_net import *
from config import *

cfg = Config('JRDB')
cfg.device_list = "0"
cfg.training_stage = 1
cfg.stage1_model_path = 'result/STAGE1_MODEL.pth'
cfg.train_backbone = True
cfg.image_size = 480, 3760
cfg.out_size = 30, 235
cfg.num_boxes = 50
cfg.num_frames = 16
cfg.use_multi_gpu=1
cfg.batch_size = 1
cfg.test_batch_size = 1
cfg.train_learning_rate = 1e-4
cfg.train_dropout_prob = 0.3
cfg.weight_decay = 1e-2
cfg.lr_plan = {2:1e-5, 12:1e-6, 26:1e-7}
cfg.max_epoch = 500
cfg.name_output='Real-time Trajectory-based_Social_Group_Detection'
cfg.exp_note = 'JRDB_stage2'
train_net_simin(cfg)
