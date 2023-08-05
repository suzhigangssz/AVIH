import os
import sys
sys.path.append(os.path.join('networks'))
from networks.CosFace import CosFace
from networks.SphereFace import SphereFace
from networks.ArcFace import ArcFace
from models_class import *


def getmodel(model, **kwargs):
	"""
		select the face model according to its name
		:param face_model: string
		:param FLAGS: a tf FLAGS (should be replace later)
		:param is_use_crop: boolean, whether the network accepted cropped images or uncropped images
		:loss_type: string, the loss to generate adversarial examples
		return:
		a model class
	"""
	img_shape = (112, 112)
	if model == 'CosFace':
		model = CosFace(**kwargs)
		img_shape = (112, 96)
	elif model == 'SphereFace':
		model = SphereFace(**kwargs)
		img_shape = (112, 96)
	elif model == 'ArcFace':
		model = ArcFace()
	elif model == 'Resnet18':
		model = ResNet18()
		img_shape = (32, 32)
		model = model.cuda()
		model = torch.nn.DataParallel(model)
		print('==> Resuming from checkpoint..')
		assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
		checkpoint2 = torch.load('./checkpoints/res.pth')
		model.load_state_dict(checkpoint2['net'])
	else:
		raise Exception
	return model, img_shape
