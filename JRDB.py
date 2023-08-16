import torch
from torch.utils import data
from PIL import Image
import numpy as np
import json
import os

seqs = ['bytes-cafe-2019-02-07_0', 'clark-center-2019-02-28_0', 'clark-center-2019-02-28_1', 'clark-center-intersection-2019-02-28_0',
        'cubberly-auditorium-2019-04-22_0', 'forbes-cafe-2019-01-22_0', 'gates-159-group-meeting-2019-04-03_0',
        'gates-ai-lab-2019-02-08_0', 'gates-basement-elevators-2019-01-17_1', 'gates-to-clark-2019-02-28_1',
        'hewlett-packard-intersection-2019-01-24_0', 'huang-2-2019-01-25_0',
        'huang-basement-2019-01-25_0', 'huang-lane-2019-02-12_0',
        'jordan-hall-2019-04-22_0', 'memorial-court-2019-03-16_0', 'meyer-green-2019-03-16_0', 'nvidia-aud-2019-04-18_0',
        'packard-poster-session-2019-03-20_0', 'packard-poster-session-2019-03-20_1', 'packard-poster-session-2019-03-20_2',
        'stlc-111-2019-04-19_0', 'svl-meeting-gates-2-2019-04-08_0', 'svl-meeting-gates-2-2019-04-08_1',
        'tressider-2019-03-16_0', 'tressider-2019-03-16_1', 'tressider-2019-04-26_2','cubberly-auditorium-2019-04-22_1',
        'discovery-walk-2019-02-28_0', 'discovery-walk-2019-02-28_1', 'food-trucks-2019-02-12_0',
                  'gates-ai-lab-2019-04-17_0', 'gates-basement-elevators-2019-01-17_0', 'gates-foyer-2019-01-17_0',
                  'gates-to-clark-2019-02-28_0',
                  'hewlett-class-2019-01-23_0', 'hewlett-class-2019-01-23_1', 'huang-2-2019-01-25_1',
                  'huang-intersection-2019-01-22_0',
                  'indoor-coupa-cafe-2019-02-06_0', 'lomita-serra-intersection-2019-01-30_0',
                  'meyer-green-2019-03-16_1', 'nvidia-aud-2019-01-25_0',
                  'nvidia-aud-2019-04-18_1', 'nvidia-aud-2019-04-18_2', 'outdoor-coupa-cafe-2019-02-06_0',
                  'quarry-road-2019-02-28_0',
                  'serra-street-2019-01-30_0', 'stlc-111-2019-04-19_1', 'stlc-111-2019-04-19_2',
                  'tressider-2019-03-16_2', 'tressider-2019-04-26_0',
                  'tressider-2019-04-26_1', 'tressider-2019-04-26_3'] #train+validation+test

FRAMES_NUM = {1: 1727, 2: 579, 3: 1440, 4: 1008, 5: 1297, 6: 1447, 7: 863, 8: 466, 9: 1010, 10: 725,
              11: 1729, 12: 873, 13: 1155, 14: 724, 15: 1523, 16: 1090, 17: 459, 18: 1013, 19: 717, 20: 852,
              21: 1372, 22: 1736, 23: 867, 24: 872, 25: 431, 26: 504, 27: 1441, 28: 1078, 29: 714, 30: 788, 31: 1603, 32: 1221, 33: 637, 34: 1662, 35: 424, 36: 790, 37: 786,
                38: 568, 39: 1658, 40: 1667, 41: 1176, 42: 1005, 43: 1226, 44: 502, 45: 497, 46: 1650, 47: 433,
                48: 1398, 49: 497, 50: 474, 51: 641, 52: 1221, 53: 1660, 54: 1658} #train+validation+test

FRAMES_SIZE = (480, 3760) #H,W
all_human_pose_action = ['walking', 'standing', 'sitting', 'cycling', 'going upstairs', 'bending', 'going downstairs', 'skating', 'scootering', 'running'] #11

pose_1 = ['walking', 'standing', 'sitting','other']
pose_2 = ['cycling', 'going upstairs', 'bending', 'other']
pose_3 = ['going downstairs', 'skating', 'scootering', 'running']

other_actions = ['holding sth', 'listening to someone', 'talking to someone','looking at robot', 'looking into sth',
           'looking at sth', 'typing', 'interaction with door', 'eating sth', 'talking on the phone',
           'reading', 'pointing at sth', 'pushing', 'greeting gestures'] #14

other_1 = ['holding sth', 'listening to someone', 'talking to someone', 'other']
other_2 = ['looking at robot', 'looking into sth', 'looking at sth', 'typing','interaction with door','eating sth', 'other']
other_3 = [ 'talking on the phone', 'reading', 'pointing at sth', 'pushing', 'greeting gestures']

POSE_ACTIONS_ID = {a: i for i, a in enumerate(all_human_pose_action)}
OTHER_ACTIONS_ID = {a: i for i, a in enumerate(other_actions)}

def JRDB_det_read_annotations(path, sid):
    H, W = 480, 3760
    annotations = {}
    path = path + '/%s.json' % (sid)
    f = json.load(open(path, 'rb'))
    for fids in sorted(f['detections'].keys()):
        fid = fids.split('.')[0]
        fid = int(fid)
        if fid >= 0 and fid <= FRAMES_NUM[seqs.index(sid)+1]:
            annotations[fid] = {
                'bboxes': [],
                'area': [],
                'score':[]
                }
            for i in range(len(f['detections'][fids])): #for each pedestrain
                x, y, w, h = f['detections'][fids][i]['box']
                score = f['detections'][fids][i]['score']
                area = w * h
                x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
                annotations[fid]['bboxes'].append((x1, y1, x2, y2))  # [0,1]
                annotations[fid]['area'].append(area)
                annotations[fid]['score'].append(score)
    return annotations

def JRDB_test_det_read_annotations(path, sid):
    H, W = 480, 3760
    annotations = {}
    path = path+'/'+sid
    frames = sorted(os.listdir(path))
    for fids in frames:
        fid = int(fids.split('.')[0])
        if fid>=0 and fid <= FRAMES_NUM[seqs.index(sid)+1]:
            annotations[fid] = {
                'bboxes': [],
                'area': [],
                'score': []
            }
            with open(path+'/'+fids, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    values = l[:-1].split(' ')
                    if float(values[-1])>=0.2:
                        x1,y1,x2,y2 = values[2:6]
                        x1, y1, x2, y2 = float(x1)/W, float(y1)/H, float(x2)/W, float(y2)/H
                        score = float(values[-1])
                        annotations[fid]['bboxes'].append((x1, y1, x2, y2))  # [0,1]
                        annotations[fid]['area'].append(abs(x2-x1)*abs(y2-y1))
                        annotations[fid]['score'].append(score)
    return annotations

def JRDB_read_annotations(path, sid):
    H, W = 480, 3760
    annotations = {}
    path = path + '/%s.json' % (sid)
    f = json.load(open(path, 'rb'))
    for fids in sorted(f['labels'].keys()):
        fid = fids.split('.')[0]
        fid = int(fid)
        if fid >= 0 and fid <= FRAMES_NUM[seqs.index(sid)+1]:
            annotations[fid] = {
                'bboxes': [],
                'label_id':[],
                'cluster_IDs': [],
                'cluster_IDs_diff': [],
                'pose_action': [],
                'pose_action_diff': [],
                'other_action': [],
                'other_action_diff': [],
                'pose_action_1': [],
                'pose_action_diff_1': [],
                'pose_action_2': [],
                'pose_action_diff_2': [],
                'pose_action_3': [],
                'pose_action_diff_3': [],
                'other_action_1': [],
                'other_action_diff_1': [],
                'other_action_2': [],
                'other_action_diff_2': [],
                'other_action_3': [],
                'other_action_diff_3': [],
                'other':[],
                'other_diff':[],
                'area': [],
                'no_eval': []}
            #_______________________________
            for i in range(len(f['labels'][fids])): #for each box
                # __INIT__________________________
                temp_pose_action_1, temp_pose_action_2, temp_pose_action_3,f_temp_pose_action,\
                temp_diff_pose_action_1, temp_diff_pose_action_2, temp_diff_pose_action_3,f_temp_diff_pose_action = [],[],[],[],[],[],[],[]
                temp_other_action_1, temp_other_action_2, temp_other_action_3,f_temp_other_action, \
                temp_diff_other_action_1, temp_diff_other_action_2, temp_diff_other_action_3, f_temp_diff_other_action = [], [], [], [], [], [], [], []

                for z in range(len(pose_1)):
                    temp_pose_action_1.append(0)
                    temp_diff_pose_action_1.append(0)

                for z in range(len(pose_2)):
                    temp_pose_action_2.append(0)
                    temp_diff_pose_action_2.append(0)

                for z in range(len(pose_3)):
                    temp_pose_action_3.append(0)
                    temp_diff_pose_action_3.append(0)

                for z in range(len(other_1)):
                    temp_other_action_1.append(0)
                    temp_diff_other_action_1.append(0)

                for z in range(len(other_2)):
                    temp_other_action_2.append(0)
                    temp_diff_other_action_2.append(0)

                for z in range(len(other_3)):
                    temp_other_action_3.append(0)
                    temp_diff_other_action_3.append(0)

                for z in range(len(all_human_pose_action)):
                    f_temp_pose_action.append(0)
                    f_temp_diff_pose_action.append(0)

                for z in range(len(other_actions)):
                    f_temp_other_action.append(0)
                    f_temp_diff_other_action.append(0)
                #_____________________________________________________
                x, y, w, h = f['labels'][fids][i]['box']
                area = w*h
                x1, y1, x2, y2 = x/W, y/H, (x + w)/W, (y + h)/H
                ID = f['labels'][fids][i]['social_group']['cluster_ID']
                ID_stat = f['labels'][fids][i]['social_group']['cluster_stat']
                label_id=f['labels'][fids][i]['label_id']
                for action_label in f['labels'][fids][i]['action_label']:
                    if action_label in all_human_pose_action:
                        f_temp_pose_action[POSE_ACTIONS_ID[action_label]] = 1
                        f_temp_diff_pose_action[POSE_ACTIONS_ID[action_label]] = f['labels'][fids][i]['action_label'][action_label]

                    elif action_label in other_actions:
                        f_temp_other_action[OTHER_ACTIONS_ID[action_label]] = 1
                        f_temp_diff_other_action[OTHER_ACTIONS_ID[action_label]] = f['labels'][fids][i]['action_label'][action_label]


                for p in range(len(all_human_pose_action)):
                    if p in [0, 1, 2] and f_temp_pose_action[p] == 1:
                        temp_pose_action_1[p] = 1
                        temp_diff_pose_action_1[p] = f_temp_diff_pose_action[p]

                    elif p not in [0, 1, 2] and f_temp_pose_action[p] == 1:
                        temp_pose_action_1[3] = 1
                        temp_diff_pose_action_1[3] = 'other' #?!

                        if p in [3, 4, 5] and f_temp_pose_action[p] == 1:
                            temp_pose_action_2[p-3] = 1
                            temp_diff_pose_action_2[p-3] = f_temp_diff_pose_action[p]

                        elif p not in [3, 4, 5] and f_temp_pose_action[p] == 1:
                            temp_pose_action_2[3] = 1
                            temp_diff_pose_action_2[3] = 'other'

                            if p in [6, 7, 8, 9] and f_temp_pose_action[p] == 1:
                                temp_pose_action_3[p-6] = 1
                                temp_diff_pose_action_3[p-6] = f_temp_diff_pose_action[p]

                for o in range(len(other_actions)):
                    if o in [0, 1, 2] and f_temp_other_action[o] == 1:
                        temp_other_action_1[o] = 1
                        temp_diff_other_action_1[o] = f_temp_diff_other_action[o]

                    elif o not in [0, 1, 2] and f_temp_other_action[o] == 1:
                        temp_other_action_1[3] = 1
                        temp_diff_other_action_1[3] = 'other'

                        if o in [3, 4, 5, 6, 7, 8] and f_temp_other_action[o] == 1:
                            # temp_other = 1
                            temp_other_action_2[o-3] = 1
                            temp_diff_other_action_2[o-3] = f_temp_diff_other_action[o]

                        elif o not in [3, 4, 5,6,7,8] and f_temp_other_action[o] == 1:
                            temp_other_action_2[6] = 1
                            temp_diff_other_action_2[6] = 'other'

                            if o in [9, 10, 11, 12, 13] and f_temp_other_action[o] == 1:
                                # temp_other = 1
                                temp_other_action_3[o-9] = 1
                                temp_diff_other_action_3[o-9] = f_temp_diff_other_action[o]

                #_______________________________________________________________________________________________________
                annotations[fid]['bboxes'].append((x1, y1, x2, y2)) #[0,1]
                annotations[fid]['cluster_IDs'].append(ID)
                annotations[fid]['cluster_IDs_diff'].append(ID_stat)
                annotations[fid]['label_id'].append(label_id)
                annotations[fid]['pose_action_1'].append(temp_pose_action_1)  # 1-hot-encoding
                annotations[fid]['pose_action_2'].append(temp_pose_action_2)  # 1-hot-encoding
                annotations[fid]['pose_action_3'].append(temp_pose_action_3)  # 1-hot-encoding
                annotations[fid]['pose_action_diff_1'].append(temp_diff_pose_action_1)  # 1-hot-encoding
                annotations[fid]['pose_action_diff_2'].append(temp_diff_pose_action_2)  # 1-hot-encoding
                annotations[fid]['pose_action_diff_3'].append(temp_diff_pose_action_3)  # 1-hot-encoding
                annotations[fid]['other_action_1'].append(temp_other_action_1)  # 1-hot-encoding
                annotations[fid]['other_action_2'].append(temp_other_action_2)  # 1-hot-encoding
                annotations[fid]['other_action_3'].append(temp_other_action_3)  # 1-hot-encoding
                annotations[fid]['other_action_diff_1'].append(temp_diff_other_action_1)  # 1-hot-encoding
                annotations[fid]['other_action_diff_2'].append(temp_diff_other_action_2)  # 1-hot-encoding
                annotations[fid]['other_action_diff_3'].append(temp_diff_other_action_3)  # 1-hot-encoding
                annotations[fid]['pose_action'].append(f_temp_pose_action)  # 1-hot-encoding
                annotations[fid]['pose_action_diff'].append(f_temp_diff_pose_action)  # 1-hot-encoding
                annotations[fid]['other_action'].append(f_temp_other_action)  # 1-hot-encoding
                annotations[fid]['other_action_diff'].append(f_temp_diff_other_action)  # 1-hot-encoding
                annotations[fid]['area'].append(area)
                annotations[fid]['no_eval'].append(f['labels'][fids][i]['attributes']['no_eval'])

    return annotations

def JRDB_det_read_dataset(path, seqs, split):
    data = {}
    for sid in seqs:
        if split in ['train', 'val']:
            data[sid] = JRDB_det_read_annotations(path, sid)
        elif split == 'test':
            data[sid] = JRDB_test_det_read_annotations(path, sid)
    return data


def JRDB_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = JRDB_read_annotations(path, sid)
    return data

def JRDB_all_frames(anns): #key_frames
    temp = []
    for s in anns:
        for f in anns[s]:
            # if int(f) == 15:
            if int(f) >= 30 and (int(f) % 15) == 0:
                temp.append((s, int(f)))
    return temp

class JRDBDataset(data.Dataset):

    def __init__(self, anns, det_anns, frames, images_path, image_size, feature_size, stat):
        self.anns = anns
        self.det_anns = det_anns
        self.frames = frames
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size
        self.stat = stat
        self.area_thresh = 500

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        print("index:",index)
        select_frames = self.get_frames(self.frames[index])
        sample = self.load_samples_sequence(select_frames)
        return sample

    def get_frames(self, frame):
        sid, src_fid = frame    # sid : folder name   src_fid: frami ke entekhab shode
        print("src_fid:", src_fid)
        return [(sid, src_fid, fid) for fid in
                [src_fid - 30, src_fid - 28, src_fid - 26, src_fid - 24, src_fid - 22, src_fid - 20, src_fid - 18, src_fid - 16,
                 src_fid -14, src_fid - 12, src_fid - 10, src_fid - 8, src_fid - 6, src_fid - 4, src_fid - 2, src_fid]]

    def load_samples_sequence(self, select_frames):
        T = 16
        OH, OW = self.feature_size
        images, bboxes_num, det_box = [], [], []
        list_info_tracks=[]
        pose_action, other_action, activity, cluster_ID, pose_stat, other_action_stat, activity_stat, cluster_ID_stat = [], [], [], [], [], [], [], []
        other, other_stat = [], []
        pose_action_1, pose_action_2, pose_action_3, other_action_1,other_action_2,other_action_3=[],[],[],[],[],[]
        for i, (sid, src_fid, fid) in enumerate(select_frames):
            print(sid, fid)
            cluster_ID_key_farme={}
            Dic_persons = {}
            current_name_persons= self.anns[sid][fid]['label_id']
            current_bboxes = self.anns[sid][fid]['bboxes']
            # ----Simin----------------
            for inx, box in enumerate(current_bboxes):
                if self.anns[sid][fid]['cluster_IDs_diff'][inx] != 3:
                    Dic_persons[current_name_persons[inx]] = box
            list_info_tracks.append(Dic_persons)
            #
            curr_box, curr_det_box = [], []
        #----Simin----------------
            if i == (T-1):
                for inxz in range(len(self.anns[sid][fid]["bboxes"])):
                    if self.anns[sid][fid]["cluster_IDs_diff"][inxz]!=3:
                       cluster_ID_key_farme[self.anns[sid][fid]["label_id"][inxz]] = self.anns[sid][fid]["cluster_IDs"][inxz]

                # ----Simin----------------
                final_bboxes, score = [], []
                if self.stat=='train':
                    for x in range(len(curr_box)):
                        curr_pose_action_1, curr_pose_action_2, curr_pose_action_3 = [], [], []
                        curr_other_action_1, curr_other_action_2, curr_other_action_3 = [], [], []

                        if len(curr_pose_action_1) >0 and self.anns[sid][fid]['cluster_IDs_diff'][x] != 'Difficult':
                            cluster_ID.append(self.anns[sid][fid]['cluster_IDs'][x])
                            final_bboxes.append(curr_box[x])


                bboxes_num = len(final_bboxes) # matched detections
                num_track = bboxes_num
                keyframe_sid = sid
                keyframe_fid = fid
        #_______________________________________________________________________________________________________________

        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        final_bbox = np.array(final_bboxes, dtype=np.float).reshape(num_track, 4)
        if self.stat in ['train', 'val']:
            cluster_ID = np.array(cluster_ID, dtype=np.int32).reshape(num_track)
            cluster_ID = torch.from_numpy(cluster_ID).long()  # [num_track]

        final_bbox = torch.from_numpy(final_bbox).float()  #[num_track, 8, 4] #in feature-scale!
        bboxes_num = torch.from_numpy(bboxes_num).int()  #[num_track]

        if self.stat in ['train', 'val']:
            return list_info_tracks,images, final_bbox, cluster_ID_key_farme, bboxes_num, pose_action, pose_action_1, pose_action_2, pose_action_3,\
                   other, other_action, other_action_1, other_action_2, other_action_3, keyframe_sid, keyframe_fid
        else:
            return images, final_bbox, bboxes_num, score, keyframe_sid, keyframe_fid