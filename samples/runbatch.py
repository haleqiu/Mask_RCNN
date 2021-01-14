import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import cv2
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def write_seg(Dir, Count, R):
    seg = np.zeros(R['masks'].shape[:2])
    movable_objects = [1,2,3,4,6,8]
    for objec_idx in range(R['class_ids'].shape[0]):
        if R['class_ids'][objec_idx] in movable_objects:
            seg = np.where(np.invert(R['masks'][...,objec_idx]), seg, R['class_ids'][objec_idx])
    if not os.path.isdir(Dir):
        os.mkdir(Dir)
    cv2.imwrite(os.path.join(Dir, "%06d.png"%Count), seg)

# Load a random image from the images folder

def run_folder(file_names, model):
    for f in file_names:
        if os.path.isfile(BASE_DIR + "/rcnnseg_" + Folder + "/" + f):
            print(f + "continue")
            continue
        if not os.path.splitext(f)[-1] == ".png":
            continue
        image = skimage.io.imread(os.path.join(IMAGE_DIR, f))

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], save_path = BASE_DIR + "/mrcnn_" + Folder + "/" + f)
        write_seg(BASE_DIR + "/rcnnseg_" + Folder, int(os.path.splitext(f)[0]), r)
    

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

BASE_DIR = "/data/shibuya_640_360_fov45_few_people_bags/2020-08-29-03-56-21"
Folder = "image_0"
IMAGE_DIR = os.path.join(BASE_DIR, Folder)
file_names = next(os.walk(IMAGE_DIR))[2]
file_names.sort()
if not os.path.isdir(BASE_DIR + "/mrcnn_" + Folder):
    os.mkdir(BASE_DIR + "/mrcnn_" + Folder)

run_folder(file_names, model)

Folder = "image_1"
IMAGE_DIR = os.path.join(BASE_DIR, Folder)
file_names = next(os.walk(IMAGE_DIR))[2]
if not os.path.isdir(BASE_DIR + "/mrcnn_" + Folder):
    os.mkdir(BASE_DIR + "/mrcnn_" + Folder)

run_folder(file_names, model)

