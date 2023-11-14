
import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

import datasets.dataset_utils as dataset_utils
from scipy.spatial.transform import Rotation as R


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database",
                 queries_folder="queries", positive_dist_threshold_pos=0.2, positive_dist_threshold_rot=0.5,
                 image_size=512, resize_test_imgs=False):
        self.database_folder = dataset_folder + "/" + database_folder
        self.queries_folder = dataset_folder + "/" + queries_folder
        self.database_paths = dataset_utils.read_images_paths(self.database_folder, get_abs_path=True)
        self.queries_paths = dataset_utils.read_images_paths(self.queries_folder, get_abs_path=True)
        
        self.dataset_name = os.path.basename(dataset_folder)
        
        #### Read paths and UTM coordinates for all images.
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms_pos = np.array([(float(path.split("@")[1]) / 100, float(path.split("@")[2]) / 100) for path in self.database_paths]).astype(float)
        self.queries_utms_pos = np.array([(float(path.split("@")[1]) / 100, float(path.split("@")[2]) / 100) for path in self.queries_paths]).astype(float)

        self.database_utms_rot = np.array([R.from_euler('xyz', (path.split("@")[9], path.split("@")[10], path.split("@")[11].split('.')[0])).as_quat() for path in self.database_paths]).astype(float)
        self.queries_utms_rot = np.array([R.from_euler('xyz', (path.split("@")[9], path.split("@")[10], path.split("@")[11].split('.')[0])).as_quat() for path in self.queries_paths]).astype(float)

        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn_pos = NearestNeighbors(n_jobs=-1)
        knn_pos.fit(self.database_utms_pos)
        positives_per_query_pos = knn_pos.radius_neighbors(
            self.queries_utms_pos, radius=positive_dist_threshold_pos, return_distance=False
        )

        knn_rot = NearestNeighbors(n_jobs=-1)
        knn_rot.fit(self.database_utms_rot)
        positives_per_query_rot = knn_rot.radius_neighbors(
            self.queries_utms_rot, radius=positive_dist_threshold_rot, return_distance=False
        )

        self.positives_per_query = [a[np.in1d(a, b)] for a, b in zip(positives_per_query_pos, positives_per_query_rot)]
        
        self.images_paths = self.database_paths + self.queries_paths
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        transforms_list = []
        if resize_test_imgs:
            # Resize to image_size along the shorter side while maintaining aspect ratio
            transforms_list += [transforms.Resize(image_size, antialias=True)]
        transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        self.base_transform = transforms.Compose(transforms_list)
    
    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = TestDataset.open_image(image_path)
        normalized_img = self.base_transform(pil_img)
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"
    
    def get_positives(self):
        return self.positives_per_query
