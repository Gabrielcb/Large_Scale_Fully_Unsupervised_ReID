import torch
import torchreid
import torchvision
from torchvision.models import resnet50, densenet121, inception_v3, efficientnet_b0, vit_b_16
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
from torch.nn import Module, Dropout, BatchNorm1d, BatchNorm2d, Linear, AdaptiveAvgPool2d 
from torch.nn import CrossEntropyLoss, Sigmoid, Softmax, ReLU, AdaptiveMaxPool2d, Conv2d
from torch.nn import functional as F
from torch import nn


from torch.nn.parallel import DistributedDataParallel as DDP

#from synchronized_batchnorm_pytorch.sync_batchnorm import convert_model

import numpy as np
import warnings

import os

try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )

# VISION TRANSFORMER IS USING ONLY PERSON IMAGE DIMENSION! TO APPLY IT TO VEHICLES OR OTHER OBJECTS, YOU MUST 
# CHANGE THE IMAGE DIMENSIONS FOR 224 X 224 OR CHANGE THE CODE TO ACCEPT ARBITRARY DIMENSIONS (only on function calling)

def getDCNN(gpu_indexes, model_name, model_path=None, embedding_size=None):
	#def getDCNN(model_name, model_path=None, embedding_size=None):

	if model_name == "resnet50":

		if embedding_size is None:
			embedding_size = 2048

		# loading ResNet50
		model_source = resnet50(pretrained=True)
		model_source = ResNet50ReID(model_source)
		
		model_momentum = resnet50(pretrained=True)
		model_momentum = ResNet50ReID(model_momentum)
		
		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		'''
		gpu_id = int(os.environ["LOCAL_RANK"])
		print("Local Rank for %s:%d" % (model_name, gpu_id))
		model_source.to(gpu_id)
		model_momentum.to(gpu_id)

		model_source = DDP(model_source, device_ids=[gpu_id])
		model_momentum = DDP(model_momentum, device_ids=[gpu_id])
		'''

		
		#model_source = convert_model(model_source)
		#model_momentum = convert_model(model_momentum)

		if model_path:
			model_source.load_state_dict(torch.load(model_path)) 
		
		
		model_momentum.load_state_dict(model_source.state_dict())
		
		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()
		
		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "osnet":

		if embedding_size is None:
			embedding_size = 512

		# loading OSNet	
		model_source = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
		model_source = OSNETReID(model_source)

		model_momentum = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
		model_momentum = OSNETReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		'''
		gpu_id = int(os.environ["LOCAL_RANK"])
		print("Local Rank for %s:%d" % (model_name, gpu_id))
		model_source.to(gpu_id)
		model_momentum.to(gpu_id)

		model_source = DDP(model_source, device_ids=[gpu_id])
		model_momentum = DDP(model_momentum, device_ids=[gpu_id])
		'''

		#model_source = convert_model(model_source)
		#model_momentum = convert_model(model_momentum)

		if model_path:
			model_source.load_state_dict(torch.load(model_path)) 

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "densenet121":

		if embedding_size is None:
			embedding_size = 2048

		# loading DenseNet121
		model_source = densenet121(pretrained=True)
		model_source = DenseNet121ReID(model_source)

		model_momentum = densenet121(pretrained=True)
		model_momentum = DenseNet121ReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		'''
		gpu_id = int(os.environ["LOCAL_RANK"])
		if gpu_id == 3:
			print("Local Rank for %s:%d" % (model_name, gpu_id))
			model_source.to(gpu_id)
			model_momentum.to(gpu_id)

			model_source = DDP(model_source, device_ids=[gpu_id])
			model_momentum = DDP(model_momentum, device_ids=[gpu_id])
		'''

			#model_source = convert_model(model_source)
			#model_momentum = convert_model(model_momentum)

		if model_path:
			model_source.load_state_dict(torch.load(model_path)) 

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "inceptionV3":

		if embedding_size is None:
			embedding_size = 2048

		# Loading inceptionV3
		model_source = inception_v3(pretrained=True)
		model_source = inceptionV3ReID(model_source)

		model_momentum = inception_v3(pretrained=True)
		model_momentum = inceptionV3ReID(model_momentum)

		#model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		#model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "ViT":

		if embedding_size is None:
			embedding_size = 2048

		# Loading inceptionV3
		model_source = vit_b_16(pretrained=True)
		model_source = ViTReID(model_source, 224, 224)

		model_momentum = vit_b_16(pretrained=True)
		model_momentum = ViTReID(model_momentum, 224, 224)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif model_name == "efficientnetB0":

		if embedding_size is None:
			embedding_size = 2048

		# Loading inceptionV3
		model_source = efficientnet_b0(pretrained=True)
		model_source = efficientnetB0ReID(model_source)

		model_momentum = efficientnet_b0(pretrained=True)
		model_momentum = efficientnetB0ReID(model_momentum)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		if model_path:
			model_source.load_state_dict(torch.load(model_path)) 

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	elif "convnext" in model_name:
	
		# Loading convnext
		convnext_type = model_name.split("_")[1]

		if convnext_type == "tiny":	
			feat_dim = 768	
			model_source = convnext_tiny(pretrained=True)
			model_momentum = convnext_tiny(pretrained=True)
		elif convnext_type == "small":
			feat_dim = 768		
			model_source = convnext_small(pretrained=True)
			model_momentum = convnext_small(pretrained=True)
		elif convnext_type == "base":		
			feat_dim = 1024
			model_source = convnext_base(pretrained=True)
			model_momentum = convnext_base(pretrained=True)
		elif convnext_type == "large":		
			model_source = convnext_large(pretrained=True)
			model_momentum = convnext_large(pretrained=True)

		model_source = convnextReID(model_source, feat_dim=feat_dim)
		model_momentum = convnextReID(model_momentum, feat_dim=feat_dim)

		model_source = nn.DataParallel(model_source, device_ids=gpu_indexes)
		model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

		if model_path:
			model_source.load_state_dict(torch.load(model_path)) 

		model_momentum.load_state_dict(model_source.state_dict())

		model_source = model_source.cuda(gpu_indexes[0])
		model_source = model_source.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	return model_source, model_momentum



def getEnsembles(gpu_indexes, resnet_path=None, osnet_path=None, densenet_path=None):

	# loading ResNet50
	model_source_resnet50 = resnet50(pretrained=True)
	#model_source_resnet50 = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
	model_source_resnet50 = ResNet50ReID(model_source_resnet50)

	model_momentum_resnet50 = resnet50(pretrained=True)
	#model_momentum_resnet50 = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
	model_momentum_resnet50 = ResNet50ReID(model_momentum_resnet50)

	model_source_resnet50 = nn.DataParallel(model_source_resnet50, device_ids=gpu_indexes)
	model_momentum_resnet50 = nn.DataParallel(model_momentum_resnet50, device_ids=gpu_indexes)

	#model_source_resnet50 = nn.parallel.DistributedDataParallel(model_source_resnet50, find_unused_parameters=True, device_ids=gpu_indexes)
	#model_momentum_resnet50 = nn.parallel.DistributedDataParallel(model_momentum_resnet50, find_unused_parameters=True, device_ids=gpu_indexes)

	model_source_resnet50 = model_source_resnet50.cuda(gpu_indexes[0])
	model_momentum_resnet50 = model_momentum_resnet50.cuda(gpu_indexes[0])

	if resnet_path:
		model_source_resnet50.load_state_dict(torch.load(resnet_path))
		
	model_momentum_resnet50.load_state_dict(model_source_resnet50.state_dict())

	#model_source_resnet50 = model_source_resnet50.cuda(gpu_indexes[0])
	model_source_resnet50 = model_source_resnet50.eval()

	#model_momentum_resnet50 = model_momentum_resnet50.cuda(gpu_indexes[0])
	model_momentum_resnet50 = model_momentum_resnet50.eval()

	# loading OSNet	
	model_source_osnet = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
	model_source_osnet = OSNETReID(model_source_osnet)

	model_momentum_osnet = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
	model_momentum_osnet = OSNETReID(model_momentum_osnet)

	model_source_osnet = nn.DataParallel(model_source_osnet, device_ids=gpu_indexes)
	model_momentum_osnet = nn.DataParallel(model_momentum_osnet, device_ids=gpu_indexes)

	model_source_osnet = model_source_osnet.cuda(gpu_indexes[0])
	model_momentum_osnet = model_momentum_osnet.cuda(gpu_indexes[0])
	#model_source_osnet = nn.parallel.DistributedDataParallel(model_source_osnet, find_unused_parameters=True, device_ids=gpu_indexes)
	#model_momentum_osnet = nn.parallel.DistributedDataParallel(model_momentum_osnet, find_unused_parameters=True, device_ids=gpu_indexes)

	if osnet_path:
		model_source_osnet.load_state_dict(torch.load(osnet_path))

	model_momentum_osnet.load_state_dict(model_source_osnet.state_dict())

	#model_source_osnet = model_source_osnet.cuda(gpu_indexes[0])
	model_source_osnet = model_source_osnet.eval()

	#model_momentum_osnet = model_momentum_osnet.cuda(gpu_indexes[0])
	model_momentum_osnet = model_momentum_osnet.eval()

	# loading DenseNet121
	model_source_densenet121 = densenet121(pretrained=True)
	model_source_densenet121 = DenseNet121ReID(model_source_densenet121)

	model_momentum_densenet121 = densenet121(pretrained=True)
	model_momentum_densenet121 = DenseNet121ReID(model_momentum_densenet121)

	model_source_densenet121 = nn.DataParallel(model_source_densenet121, device_ids=gpu_indexes)
	model_momentum_densenet121 = nn.DataParallel(model_momentum_densenet121, device_ids=gpu_indexes)

	model_source_densenet121 = model_source_densenet121.cuda(gpu_indexes[0])
	model_momentum_densenet121 = model_momentum_densenet121.cuda(gpu_indexes[0])
	#model_source_densenet121 = nn.parallel.DistributedDataParallel(model_source_densenet121, find_unused_parameters=True, device_ids=gpu_indexes)
	#model_momentum_densenet121 = nn.parallel.DistributedDataParallel(model_momentum_densenet121, find_unused_parameters=True, device_ids=gpu_indexes)

	if densenet_path:
		model_source_densenet121.load_state_dict(torch.load(densenet_path))

	model_momentum_densenet121.load_state_dict(model_source_densenet121.state_dict())

	#model_source_densenet121 = model_source_densenet121.cuda(gpu_indexes[0])
	model_source_densenet121 = model_source_densenet121.eval()

	#model_momentum_densenet121 = model_momentum_densenet121.cuda(gpu_indexes[0])
	model_momentum_densenet121 = model_momentum_densenet121.eval()

	return model_source_resnet50, model_momentum_resnet50, model_source_osnet, model_momentum_osnet, model_source_densenet121, model_momentum_densenet121
	


## New Model Definition for ResNet50
class ResNet50ReID(Module):
    
	def __init__(self, model_base):
		super(ResNet50ReID, self).__init__()


		self.conv1 = model_base.conv1
		self.bn1 = model_base.bn1
		self.relu = model_base.relu
		self.maxpool = model_base.maxpool
		self.layer1 = model_base.layer1
		self.layer2 = model_base.layer2
		self.layer3 = model_base.layer3
		self.layer4 = model_base.layer4

		self.layer4[0].conv2.stride = (1,1)
		self.layer4[0].downsample[0].stride = (1,1)

		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		#self.expoents = torch.nn.Parameter(torch.ones(2048,1,1))
		self.last_bn = BatchNorm1d(2048)

	
	def forward(self, x, multipart=False):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		if multipart:
			x_upper = x[:,:,:8,:]
			x_middle = x[:,:,4:12,:]
			x_lower = x[:,:,8:,:]

			# Upper part pooling
			x_avg_upper = self.global_avgpool(x_upper)
			x_max_upper = self.global_maxpool(x_upper)
			x_upper = x_avg_upper + x_max_upper
		
			x_upper = x_upper.view(x_upper.size(0), -1)
			output_upper = self.last_bn(x_upper)

			# Middle part pooling
			x_avg_middle = self.global_avgpool(x_middle)
			x_max_middle = self.global_maxpool(x_middle)
			x_middle = x_avg_middle + x_max_middle
		
			x_middle = x_middle.view(x_middle.size(0), -1)
			output_middle = self.last_bn(x_middle)

			# Lower part pooling
			x_avg_lower = self.global_avgpool(x_lower)
			x_max_lower = self.global_maxpool(x_lower)
			x_lower = x_avg_lower + x_max_lower
		
			x_lower = x_lower.view(x_lower.size(0), -1)
			output_lower = self.last_bn(x_lower)

			# Whole Body pooling
			x_avg = self.global_avgpool(x)
			x_max = self.global_maxpool(x)
			x = x_avg + x_max
		
			x = x.view(x.size(0), -1)
			output = self.last_bn(x)

			return output_upper, output_middle, output_lower, output

		# attention module
		'''
		b = torch.flatten(x, start_dim=2).T
		b_norm = b/torch.norm(b,dim=1,keepdim=True)
		gap = torch.mean(x,dim=(2,3))
		gap = gap/torch.norm(gap,dim=1,keepdim=True)
		res = b_norm*gap.T
		sum_res = torch.sum(res, dim=1)
		sum_exp = torch.sum(torch.exp(sum_res), dim=0, keepdim=True)
		sum_res = torch.exp(sum_res)/sum_exp

		batch_size, num_channels, height, width = x.shape
		attention_map = sum_res.T.reshape(batch_size,1,height,width)
		x = x*attention_map
		x = torch.sum(x, dim=(2,3),keepdim=True)
		'''
		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max

		#x = torch.pow(x.clamp(min=1e-6), self.expoents)
		#x_avg = self.global_avgpool(x)
		#x = torch.pow(x_avg, 1/self.expoents)

		#x = x_max
		x = x.view(x.size(0), -1)

		output = self.last_bn(x)
		return output


class GeM(Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


## New Model Definition for ResNet50
class ResNet50SegReID(Module):
    
	def __init__(self, model_base):
		super(ResNet50SegReID, self).__init__()


		self.conv1 = model_base.conv1
		self.bn1 = model_base.bn1
		self.relu = model_base.relu
		self.maxpool = model_base.maxpool
		self.layer1 = model_base.layer1
		self.layer2 = model_base.layer2
		self.layer3 = model_base.layer3
		self.layer4 = model_base.layer4

		self.layer4[0].conv2.stride = (1,1)
		self.layer4[0].downsample[0].stride = (1,1)

		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(2048)

	
	def forward(self, x, seg_mask=None):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		# attention module
		if seg_mask is not None:
			x = x*seg_mask

		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		#x = x_max
		x = x.view(x.size(0), -1)

		output = self.last_bn(x)
		return output

## New Model Definition for DenseNet121
class DenseNet121ReID(Module):
    
	def __init__(self, model_base):
		super(DenseNet121ReID, self).__init__()

		self.model_base = model_base.features
		self.gap = AdaptiveAvgPool2d(1)
		self.gmp = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(2048)

	def forward(self, x):
		
		x = self.model_base(x)
		x = F.relu(x, inplace=True)

		x_avg = self.gap(x)
		x_max = self.gmp(x)
		x = x_avg + x_max
		x = torch.cat([x,x], dim=1)

		x = x.view(x.size(0), -1)

		output = self.last_bn(x)
		return output 

## New Definition for OSNET
class OSNETReID(Module):
    
	def __init__(self, model_base, embedding_size=512):
		super(OSNETReID, self).__init__()

		self.conv1 = model_base.conv1
		self.maxpool = model_base.maxpool
		self.conv2 = model_base.conv2
		self.conv3 = model_base.conv3
		self.conv4 = model_base.conv4
		self.conv5 = model_base.conv5
		self.avgpool = model_base.global_avgpool
		self.maxpool02 = AdaptiveMaxPool2d(output_size=(1, 1))
		#self.fc = model_base.fc
		self.last_bn = BatchNorm1d(512)


	def forward(self, x):
		
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)

		v_avg = self.avgpool(x)
		v_max = self.maxpool02(x)
		v = v_avg + v_max
		v = v.view(v.size(0), -1)
		output = self.last_bn(v)
		#output = self.fc(v)
		return output

class inceptionV3ReID(Module):
    
	def __init__(self, model_base, embedding_size=2048):
		super(inceptionV3ReID, self).__init__()


		self.Conv2d_1a_3x3 = model_base.Conv2d_1a_3x3
		self.Conv2d_2a_3x3 = model_base.Conv2d_2a_3x3
		self.Conv2d_2b_3x3 = model_base.Conv2d_2b_3x3 
		self.maxpool1 = model_base.maxpool1
		self.Conv2d_3b_1x1 = model_base.Conv2d_3b_1x1
		self.Conv2d_4a_3x3 = model_base.Conv2d_4a_3x3
		self.maxpool2 = model_base.maxpool2
		self.Mixed_5b = model_base.Mixed_5b
		self.Mixed_5c = model_base.Mixed_5c
		self.Mixed_5d = model_base.Mixed_5d
		self.Mixed_6a = model_base.Mixed_6a
		self.Mixed_6b = model_base.Mixed_6b
		self.Mixed_6c = model_base.Mixed_6c
		self.Mixed_6d = model_base.Mixed_6d
		self.Mixed_6e = model_base.Mixed_6e
		self.Mixed_7a = model_base.Mixed_7a
		self.Mixed_7b = model_base.Mixed_7b
		self.Mixed_7c = model_base.Mixed_7c
		
		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(2048)
		

	def forward(self, x):

		x = self.Conv2d_1a_3x3(x)
		# N x 32 x 149 x 149
		x = self.Conv2d_2a_3x3(x)
		# N x 32 x 147 x 147
		x = self.Conv2d_2b_3x3(x)
		# N x 64 x 147 x 147
		x = self.maxpool1(x)
		# N x 64 x 73 x 73
		x = self.Conv2d_3b_1x1(x)
		# N x 80 x 73 x 73
		x = self.Conv2d_4a_3x3(x)
		# N x 192 x 71 x 71
		x = self.maxpool2(x)
		# N x 192 x 35 x 35
		x = self.Mixed_5b(x)
		# N x 256 x 35 x 35
		x = self.Mixed_5c(x)
		# N x 288 x 35 x 35
		x = self.Mixed_5d(x)
		# N x 288 x 35 x 35
		x = self.Mixed_6a(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6b(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6c(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6d(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6e(x)
		# N x 768 x 17 x 17
		x = self.Mixed_7a(x)
		# N x 1280 x 8 x 8
		x = self.Mixed_7b(x)
		# N x 2048 x 8 x 8
		x = self.Mixed_7c(x)
		# N x 2048 x 8 x 8
		# Adaptive average pooling
		
		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		x = x.view(x.size(0), -1)

		output = self.last_bn(x)
		return output


## New Definition for Vision Transform
class ViTReID(Module):

	def __init__(self, model_base, img_height, img_width, patch_size=16, stride_size=16):
		super(ViTReID, self).__init__()

		self.patch_size = patch_size
		self.image_size = model_base.image_size
		self.hidden_dim = model_base.hidden_dim
		self.class_token = model_base.class_token

		self.conv_proj = model_base.conv_proj
		self.encoder = model_base.encoder
		self.heads = model_base.heads

		if img_height != 224 or img_width != 224:
			seq_length = (img_height // patch_size) * (img_width // patch_size)
			seq_length += 1

			self.encoder.pos_embedding = nn.Parameter(torch.empty(1, seq_length, self.hidden_dim).normal_(std=0.02))

		self.last_bn = BatchNorm1d(768)

	def _process_input(self, x):
		n, c, h, w = x.shape
		p = self.patch_size
		#torch._assert(h == self.image_size, "Wrong image height!")
		#torch._assert(w == self.image_size, "Wrong image width!")
		n_h = h // p
		n_w = w // p

		# (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
		x = self.conv_proj(x)
		# (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
		x = x.reshape(n, self.hidden_dim, n_h * n_w)

		# (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
		# The self attention layer expects inputs in the format (N, S, E)
		# where S is the source sequence length, N is the batch size, E is the
		# embedding dimension
		x = x.permute(0, 2, 1)

		return x


	def forward(self, x):

		# Reshape and permute the input tensor
		x = self._process_input(x)
		n = x.shape[0]

		# Expand the class token to the full batch
		batch_class_token = self.class_token.expand(n, -1, -1)
		x = torch.cat([batch_class_token, x], dim=1)

		x = self.encoder(x)

		# Classifier "token" as used by standard language architectures
		x = x[:, 0]

		#x = self.heads(x)
		x = self.last_bn(x)
		return x

class efficientnetB0ReID(Module):

	def __init__(self, model_base):
		super(efficientnetB0ReID, self).__init__()

		self.features = model_base.features
		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(1280)
	
	def forward(self, x):
		x = self.features(x)
		
		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		
		x = x.view(x.size(0), -1)

		output = self.last_bn(x)
		return output

class convnextReID(Module):

	def __init__(self, model_base, feat_dim=1024):
		super(convnextReID, self).__init__()

		self.features = model_base.features
		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(feat_dim)

	def forward(self, x):
		x = self.features(x)
		x_avg = self.global_avgpool(x)
		x_max = self.global_maxpool(x)
		x = x_avg + x_max
		x = x.view(x.size(0), -1)
		
		output = self.last_bn(x)
		return output