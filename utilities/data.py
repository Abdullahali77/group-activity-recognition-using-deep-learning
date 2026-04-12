import cv2
import pickle
import torch
import numpy as np
from pathlib import Path
import sys
from torch.utils.data import Dataset

activities_labels = {
    "person": {class_name.lower(): i for i, class_name in enumerate(["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"])},
    "group": {class_name: i for i, class_name in enumerate(["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"])}
}

class Hierarchical_Group_Activity_DataSet(Dataset):
    def __init__(self, videos_path, annot_path, split, labels, transform=None):
        self.videos_path = Path(videos_path)
        self.transform = transform
        self.labels = labels
        
        with open(annot_path, 'rb') as f:
            videos_annot = pickle.load(f)
            
        self.data = []
        for clip_id in split:
            clip_dirs = videos_annot[str(clip_id)]
            for clip_dir in clip_dirs.keys():
                category = clip_dirs[str(clip_dir)]['category']
                dir_frames = list(clip_dirs[str(clip_dir)]['frame_boxes_dct'].items())
                
                frames_data = []
                for frame_id, boxes in dir_frames:
                    frame_path = f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg"
                    frames_data.append((frame_path, boxes))
                
                self.data.append({'frames_data': frames_data, 'category': category})      

    def __len__(self):
        return len(self.data)
    
    def extract_person_crops(self, frame, boxes):
        crops, order, person_frame_labels = [], [], []
        for box in boxes:
            x_min, y_min, x_max, y_max = box.box
            x_center = (x_min + x_max) / 2
            person_crop = frame[max(0, y_min):y_max, max(0, x_min):x_max]
            
            if self.transform:
                person_crop = self.transform(image=person_crop)['image']

            person_label = torch.zeros(len(self.labels['person']))
            person_label[self.labels['person'][box.category]] = 1    
            
            crops.append(person_crop)
            order.append(x_center)
            person_frame_labels.append(person_label)
        
        return crops, order, person_frame_labels
                    
    def __getitem__(self, idx):
        sample = self.data[idx]
        group_label = torch.zeros(len(self.labels['group'])) 
        group_label[self.labels['group'][sample['category']]] = 1

        clip, group_labels, person_labels = [], [], []
    
        for frame_path, boxes in sample['frames_data']:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # CRITICAL: Convert BGR to RGB
            
            crops, order, frame_labels = self.extract_person_crops(frame, boxes) 
            
            # Sort left-to-right
            sorted_pairs = sorted(zip(order, crops, frame_labels), key=lambda pair: pair[0])
            sorted_crops = [c for _, c, _ in sorted_pairs]
            sorted_labels = [l for _, _, l in sorted_pairs]
                        
            crops = torch.stack(sorted_crops) if sorted_crops else torch.empty(0)
            sorted_labels = torch.stack(sorted_labels) if sorted_labels else torch.empty(0)
            
            person_labels.append(sorted_labels) 
            clip.append(crops)
            group_labels.append(group_label)
    
        # Shape: (Num_Frames, Num_People, C, H, W) -> (Num_People, Num_Frames, C, H, W)
        clip = torch.stack(clip).permute(1, 0, 2, 3, 4) 
        group_labels = torch.stack(group_labels)
        person_labels = torch.stack(person_labels).permute(1, 0, 2) 

        return clip, person_labels, group_labels

def collate_fn(batch):
    clips, person_labels, group_labels = zip(*batch)  
    max_bboxes = 12  
    padded_clips, padded_person_labels = [], []

    for clip, label in zip(clips, person_labels):
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            label_padding = torch.zeros((max_bboxes - num_bboxes, label.size(1), label.size(2)))
            clip = torch.cat((clip, clip_padding), dim=0)
            label = torch.cat((label, label_padding), dim=0)
            
        padded_clips.append(clip)
        padded_person_labels.append(label)
    
    padded_clips = torch.stack(padded_clips)
    padded_person_labels = torch.stack(padded_person_labels)
    group_labels = torch.stack(group_labels)
    
    # Use only the last frame's label for the loss
    group_labels = group_labels[:, -1, :] 
    padded_person_labels = padded_person_labels[:, :, -1, :]  
    
    b, bb, num_class = padded_person_labels.shape
    padded_person_labels = padded_person_labels.view(b * bb, num_class)

    return padded_clips, padded_person_labels, group_labels

class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated

sys.modules['boxinfo'] = sys.modules[__name__]        
