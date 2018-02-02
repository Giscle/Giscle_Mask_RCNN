import os
import json
import numpy as np
import time
import pickle
import pandas as pd
from PIL import Image, ImageDraw
import math
import random
import cv2

from config import Config
import utils

class CityscapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 40  # background + 40 labels
    
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 2048
    
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 10
    TRAIN_ROIS_PER_IMAGE = 512
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    
    
config = CityscapesConfig()
config.display()

class CityscapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """


    def load_cityscapes(self, source):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        
        images = pickle.load(open("image_{}.pickle".format(source), "rb"))
        images = np.random.permutation(images)
        labels = ['bicycle', 'bicyclegroup', 'bridge', 'building', 'bus', 'car',
                  'caravan', 'cargroup', 'dynamic', 'ego vehicle', 'fence', 'ground',
                  'guard rail', 'license plate', 'motorcycle', 'motorcyclegroup',
                  'out of roi', 'parking', 'person', 'persongroup', 'pole',
                  'polegroup', 'rail track', 'rectification border', 'rider',
                  'ridergroup', 'road', 'sidewalk', 'sky', 'static', 'terrain',
                  'traffic light', 'traffic sign', 'trailer', 'train', 'truck',
                  'truckgroup', 'tunnel', 'vegetation', 'wall']
        # Add classes
        for i, class_name in enumerate(labels):
            self.add_class("cityscapes", i + 1, class_name)

        # Add images
        for i, path in enumerate(images):
            self.add_image("cityscapes", image_id=i, path=path)
            
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cityscapes":
            return info
        else:
            super(self.__class__).image_reference(self, image_id)
            
    def load_mask(self, image_id):
        image_path = self.image_info[image_id]['path']
        json_path = image_path.replace("leftImg8bit","gtFine")
        json_path = json_path.replace(".png", "_polygons.json")

        dct = json.load(open(json_path,'r'))
        num_instances = len(dct['objects'])
        
        width = 2048
        height = 1024

        instance_masks = []
        class_ids = []

        for i in range(num_instances):
            instance = dct['objects'][i]
            class_name = instance['label']
            for c in self.class_info:
                if c['name'] == class_name:
                    class_ids.append(c['id'])
                    break
            poly = instance['polygon']
            polygon = [tuple(pt) for pt in poly]
            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            instance_masks.append(img)
            
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CityscapesDataset, self).load_mask(image_id)
            
