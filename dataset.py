
from JRDB import *
import pickle

def return_dataset(cfg):

    if cfg.dataset_name == 'JRDB':

        train_anns = JRDB_read_dataset(cfg.train_annot_path, cfg.train_seqs)
        train_det_anns = JRDB_det_read_dataset(cfg.train_det_annot_path, cfg.train_seqs, split='train')
        train_frames = JRDB_all_frames(train_anns)

        valid_anns = JRDB_read_dataset(cfg.train_annot_path, cfg.valid_seqs)
        valid_frames = JRDB_all_frames(valid_anns)
        valid_det_anns = JRDB_det_read_dataset(cfg.train_det_annot_path, cfg.valid_seqs, split='val')

        # test_anns = JRDB_read_dataset(cfg.test_annot_path, cfg.test_seqs)
        # test_det_anns = JRDB_det_read_dataset(cfg.test_det_annot_path, cfg.test_seqs, split='test')
        # test_frames = JRDB_all_frames(test_anns)
        # ein miad ham etellaate training ro yeki mikone
        training_set = JRDBDataset(train_anns, train_det_anns, train_frames, cfg.train_data_path, cfg.image_size, cfg.out_size, stat='train')
        validation_set = JRDBDataset(valid_anns, valid_det_anns, valid_frames, cfg.train_data_path, cfg.image_size, cfg.out_size, stat='val')
        # test_set = JRDBDataset(test_anns,test_det_anns, test_frames, cfg.test_data_path, cfg.image_size, cfg.out_size, stat='test')

    else:
        assert False

    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    # print('%d test samples'%len(test_frames))

    # return training_set, validation_set, test_set
    return training_set, validation_set
