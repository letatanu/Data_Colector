from torch.utils.data import Dataset
import torch
import os
import cv2
import numpy as np
class DB_Loader(Dataset):
    def __init__(self, root_dir, db):
        self.db = db
        self.root_dir = root_dir

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.path.join(self.root_dir, list(self.db.keys())[idx])
        img = cv2.imread(image_name, flags=cv2.IMREAD_COLOR)
        scale_percent = 60  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        image = np.array(resized, dtype=np.float)/255.0

        image = image.transpose((2,0,1))
        '''
        TODO: This function converts string label to int value.
        '''

        ####################----------------------###############
        label = list(self.db.values())[idx]
        return image, label