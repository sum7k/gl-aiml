import os

import tensorflow
print(tensorflow.__version__)

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import pydicom
import numpy as np
import cv2


ROOT_DIR = '/home/ubuntu/data'
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')


class DetectorConfig(Config):
	"""Configuration for training pneumonia detection on the RSNA pneumonia dataset.
	Overrides values in the base Config class.
	"""
	
	NAME = 'pneumonia'
	
	# Train on 1 GPU and 8 images per GPU. We can put multiple images on each
	# GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
	GPU_COUNT = 2
	IMAGES_PER_GPU = 8
	
	BACKBONE = 'resnet50'
	
	NUM_CLASSES = 2  # background + 1 pneumonia classes
	
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 256
	RPN_ANCHOR_SCALES = (32, 64, 128, 256)
	TRAIN_ROIS_PER_IMAGE = 32
	MAX_GT_INSTANCES = 3
	DETECTION_MAX_INSTANCES = 3
	DETECTION_MIN_CONFIDENCE = 0.7
	DETECTION_NMS_THRESHOLD = 0.1

	STEPS_PER_EPOCH = 100
	
class InferenceConfig(DetectorConfig):
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

inference_config = InferenceConfig()


class Model:
	model = None

	def __init__(self):
		print("in Init")

	def initialize(self):
		#initialize the model
		self.model = modellib.MaskRCNN(mode='inference',config=inference_config,model_dir=MODEL_DIR)
		self.model.load_weights('/home/ubuntu/data/logs/pneumonia20201217T0756/mask_rcnn_pneumonia_0014.h5', by_name = True)
		self.model.keras_model._make_predict_function()
		pass

	def load_image(self, filepath):
		data = pydicom.read_file(filepath,  force=True)
		image = data.pixel_array
		if len(image.shape) != 3 or image.shape[2] != 3:
			image = np.stack((image,) * 3, -1)
		image, window, scale, padding, crop = utils.resize_image(image,
			min_dim = inference_config.IMAGE_MIN_DIM,
			min_scale = inference_config.IMAGE_MIN_SCALE,
			max_dim = inference_config.IMAGE_MAX_DIM,
			mode = inference_config.IMAGE_RESIZE_MODE)
		return image

	def predict(self, filepath):
		image = self.load_image(filepath)
		result = self.model.detect([image])[0]
		# result = {}
		# result['rois'] = [[110, 58, 182, 100]]
		# result['class_ids'] = [[1]]
		# result['scores'] = [[.92]]
		# result['masks'] = None
		return image, result




					# dicom = request.files['dicom']

			#	 data = pydicom.read_file(filepath)
			#	 image = data.pixel_array
			#	 if len(image.shape) != 3 or image.shape[2] != 3:
			#			 image = np.stack((image,) * 3, -1)
			#	 image, window, scale, padding, crop = utils.resize_image(image,
			#															  min_dim = inference_config.IMAGE_MIN_DIM,
			#															  min_scale = inference_config.IMAGE_MIN_SCALE,
			#															  max_dim = inference_config.IMAGE_MAX_DIM,
			#															  mode = inference_config.IMAGE_RESIZE_MODE)
						
			#	 result = model.model.detect([image])[0]
			#	 print(result)




