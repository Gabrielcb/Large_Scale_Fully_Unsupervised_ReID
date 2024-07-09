import os
import copy

import torch
import torchreid
import torchvision

from torchvision.models import resnet50, densenet121, inception_v3
from torch.nn import functional as F

import numpy as np
import time
import argparse
import joblib

from Encoders import getDCNN, getEnsembles
from validateModels import validateOnDatasets
from datasetUtils import get_dataset_samples_and_statistics

from random import shuffle
from termcolor import colored
from tabulate import tabulate
from torch.backends import cudnn

from torchvision.transforms import RandomCrop, functional, GaussianBlur
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize, Resize, RandomErasing, RandomHorizontalFlip, ColorJitter, RandomGrayscale

from torch import nn
from torch.nn import Module, BatchNorm1d, Linear, AdaptiveAvgPool2d, ReLU, AdaptiveMaxPool2d, MultiheadAttention
from torch.utils.data import Dataset, DataLoader

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

def main(gpu_ids, uccs_cluster, model_name, model_path, multiview, base_lr, batch_size, lambda_bt, number_of_epoches, targets, 
									segmentation_masks_dir, kind_of_transform, path_to_train_event, dir_to_save, 
																					dir_to_save_metrics, version, eval_freq):

	print(gpu_ids)
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

	num_gpus = torch.cuda.device_count()
	print("Num GPU's:", num_gpus)

	#if len(gpu_ids) > 1:
	gpu_indexes = np.arange(num_gpus).tolist()
	#else:
	#	gpu_indexes = [0]

	print("Allocated GPU's for model:", gpu_indexes)
	
	#if multiview:	
		# loading ResNet50
	#	model_online = resnet50(pretrained=True)
	#	model_online = ResNet50ReID(model_online)

	#	model_momentum = resnet50(pretrained=True)
	#	model_momentum = ResNet50ReID(model_momentum)

	#	model_online = nn.DataParallel(model_online, device_ids=gpu_indexes)
	#	model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

	#	model_momentum.load_state_dict(model_online.state_dict())

	#	model_online = model_online.cuda(gpu_indexes[0])
	#	model_online = model_online.eval()

	#	model_momentum = model_momentum.cuda(gpu_indexes[0])
	#	model_momentum = model_momentum.eval()
		
	#else:

	model_online, model_momentum = getDCNN(gpu_indexes, model_name)
	print(model_online)

	#model_online = resnet50(pretrained=True)
	#model_online = ResNet50ReID(model_online)

	#model_momentum = resnet50(pretrained=True)
	#model_momentum = ResNet50ReID(model_momentum)

	#model_online = nn.DataParallel(model_online, device_ids=gpu_indexes)
	#model_momentum = nn.DataParallel(model_momentum, device_ids=gpu_indexes)

	#model_momentum.load_state_dict(model_online.state_dict())

	#model_online = model_online.cuda(gpu_indexes[0])
	#model_online = model_online.eval()

	#model_momentum = model_momentum.cuda(gpu_indexes[0])
	#model_momentum = model_momentum.eval()

	if model_path:
		model_online.load_state_dict(torch.load(model_path))
		model_momentum.load_state_dict(model_online.state_dict())


		model_online = model_online.cuda(gpu_indexes[0])
		model_online = model_online.eval()

		model_momentum = model_momentum.cuda(gpu_indexes[0])
		model_momentum = model_momentum.eval()

	#### Load target datasets ####
	if targets:
		targets_names = targets.split(',')
		train_images_target, gallery_images_target, queries_images_target = get_dataset_samples_and_statistics(targets_names, uccs_cluster)
		validateOnDatasets(targets_names, queries_images_target, gallery_images_target, model_name, model_online, gpu_indexes)


	if path_to_train_event:
		train_images_target = []

		for img_path in os.listdir(path_to_train_event):
			full_image_path = os.path.join(path_to_train_event, img_path)
			train_images_target.append([full_image_path, -1, -1, 'person'])
			
		train_images_target = np.array(train_images_target)

	print("Training dimension", train_images_target.shape)
	
	cmc_progress = []
	mAP_progress = []
	lr_values = []

	base_lr_values01 = np.linspace(base_lr, base_lr, num=number_of_epoches)
	base_lr_values02 = np.linspace(base_lr, base_lr, num=30)
	base_lr_values03 = np.linspace(base_lr, base_lr, num=10)
	#base_lr_values = np.concatenate((base_lr_values01, base_lr_values02, base_lr_values03))
	base_lr_values = base_lr_values01

	optimizer = torch.optim.Adam(model_online.parameters(), lr=base_lr, weight_decay=1.5e-6)
	
	total_feature_extraction_reranking_time = 0
	total_clustering_time = 0
	total_finetuning_time = 0

	dropout_prob = 0.0
	lambda_sim = 0
	eps = 1e-6
	
	dataSubset = sample(train_images_target, segmentation_masks_dir, kind_of_transform=kind_of_transform)
	loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, 
												pin_memory=True, drop_last=True, collate_fn=collate_fn)

	t0_pipeline = time.time()
	for pipeline_iter in range(1, number_of_epoches+1):

		# Training
		t0 = time.time()
		print("###============ Iteration number %d/%d ============###" % (pipeline_iter, number_of_epoches))
		
		lr_value = base_lr_values[pipeline_iter-1]
		lr_values.append(lr_value)

		print(colored("Learning Rate: %f" % lr_value, "cyan"))
		lambda_lr_warmup(optimizer, lr_value)

		#if pipeline_iter == 1:
		#	model_online.module.attetion_activated = True

		model_online.train()
		freezeLayers(model_name, model_online)

		total_loss = 0
		total_global_loss = 0
		total_spatial_loss = 0
		total_channel_loss = 0

		iteration_weights_sum = 0.0

		loss_invariance = 0
		loss_redundacy = 0

		gradients = []


		for batch_idx, batch in enumerate(loader):
			
			#fvs01 = model_online(batch[0].cuda())
			#fvs02 = model_online(batch[1].cuda())
			full_batch = torch.cat((batch[0], batch[1]), dim=0).cuda()
			fvs = model_online(full_batch)

			fvs01 = fvs[:batch_size]
			fvs02 = fvs[batch_size:]
			
			loss, batch_loss_invariance, batch_loss_redundacy = BarlowTwinsLoss(fvs01, fvs02, lambda_bt)

			loss_invariance += batch_loss_invariance
			loss_redundacy += batch_loss_redundacy
			#print("Loss: %.5f" % loss.item())
			total_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# Decay term
			weights_sum = 0
			for param in model_online.parameters():
				weights_sum += param.pow(2).sum().item()

			iteration_weights_sum += weights_sum
	
		total_loss = total_loss/(batch_idx+1)
		loss_invariance = loss_invariance/(batch_idx+1)
		loss_redundacy = loss_redundacy/(batch_idx+1)
		iteration_weights_sum = iteration_weights_sum/(batch_idx+1)

		total_global_loss = total_global_loss/(batch_idx+1)
		total_spatial_loss = total_spatial_loss/(batch_idx+1)
		total_channel_loss = total_channel_loss/(batch_idx+1)	

		print("Mean Loss: %.5f, Mean invariance loss: %.5f, Mean redundacy loss: %.5f, Mean Weights Sum: %.5f" % (total_loss, loss_invariance, 
																													loss_redundacy, iteration_weights_sum))
		
		tf = time.time()
		dt_finetuning = tf - t0
		total_finetuning_time += dt_finetuning

		# Checking reliability
		'''
		train_fvs = extractFeatures(train_images_target, model_online, 500, gpu_index=gpu_indexes[0])
		train_fvs = train_fvs/torch.norm(train_fvs, dim=1, keepdim=True)

		selected_xi = 0.05
		k = 30

		print(colored("Xi value on OPTICS clustering: %f" % selected_xi, "cyan"))
		selected_images, pseudo_labels, ratio_of_reliable_data = GPUOPTICS(train_fvs, train_images_target, k, minPts=5, xi=selected_xi)
		'''

		if pipeline_iter % eval_freq == 0:

			cmc, mAP, distmat = validateOnDatasets(targets_names, queries_images_target, gallery_images_target, model_name, 
																									model_online, gpu_indexes)
			
			cmc_progress.append(cmc)
			mAP_progress.append(mAP)

			#joblib.dump(cmc_progress, "%s/CMC_%s_%s" % (dir_to_save_metrics, model_name, version))
			#joblib.dump(mAP_progress, "%s/mAP_%s_%s" % (dir_to_save_metrics, model_name, version))
		
		#torch.save(model_online.state_dict(), "%s/model_online_%s_%s_%d.h5" % (dir_to_save, model_name, version, pipeline_iter))
		torch.save(model_online.state_dict(), "%s/model_online_%s_%s.h5" % (dir_to_save, model_name, version))
		#torch.save(model_momentum.state_dict(), "%s/model_momentum_%s_%s.h5" % (dir_to_save, model_name, version))

		#joblib.dump(reliable_data_progress, "%s/reliability_progress_%s_%s" % (dir_to_save_metrics, model_name, version))
		#joblib.dump(lr_values, "%s/lr_progress_%s" % (dir_to_save_metrics, version))

		#joblib.dump(progress_loss, "%s/loss_progress_%s_%s_%s" % (dir_to_save_metrics, "To" + target, model_name, version))
		#joblib.dump(number_of_clusters, "%s/number_clusters_%s_%s_%s" % (dir_to_save_metrics, "To" + target, model_name, version))


	#train_fvs = extractFeatures(train_images_target, model_online, 500, gpu_index=gpu_indexes[0])
	#train_fvs = train_fvs/torch.norm(train_fvs, dim=1, keepdim=True)
	#k = 100
	#minPts = 5
	#defineEpsForDBSCAN(train_fvs, k, minPts, verbose=1)

	tf_pipeline = time.time()
	total_pipeline_time = tf_pipeline - t0_pipeline	
	mean_finetuning_time = total_finetuning_time/number_of_epoches

	print("Mean Finetuning Time: %f" % mean_finetuning_time)
	print("Total pipeline Time:  %f" % total_pipeline_time)


def BarlowTwinsLoss(fvs01, fvs02, lambda_bt, eps=1e-7):

	fvs01 = (fvs01 - fvs01.mean(dim=0, keepdims=True))/(fvs01.std(dim=0, keepdims=True) + eps)
	fvs02 = (fvs02 - fvs02.mean(dim=0, keepdims=True))/(fvs02.std(dim=0, keepdims=True) + eps)

	#print(torch.norm(fvs01, dim=1, keepdim=True), torch.norm(fvs02, dim=1, keepdim=True))

	N, d = fvs01.shape
	identity_dim = torch.eye(d).cuda()
	
	C = torch.mm(fvs01.T, fvs02)/N
	
	C_diff = (C - identity_dim).pow(2)
	loss = (C_diff*(identity_dim  + lambda_bt*(1 - identity_dim))).sum()

	loss_invariance = (C_diff*identity_dim).sum().item()
	loss_redundacy = (C_diff*(1 - identity_dim)).sum().item()

	return loss, loss_invariance, loss_redundacy


def collate_fn(batch):

	images_augmented01 = []
	images_augmented02 = []

	for pair_idx in range(len(batch)):
		aug01, aug02 = batch[pair_idx][0], batch[pair_idx][1]

		images_augmented01.append(aug01)
		images_augmented02.append(aug02)

	images_augmented01 = torch.cat(tuple(images_augmented01), dim=0)
	images_augmented02 = torch.cat(tuple(images_augmented02), dim=0)
		
	return images_augmented01, images_augmented02

def lambda_lr_warmup(optimizer, lr_value):

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr_value

def load_multiple_datasets(targets_names):

	# Note that we will concatenate all training images but NOT the gallery and query images,
	# since we would like to have an unique training set but separated evaluations

	train_images_target = []
	gallery_images_target = []
	queries_images_target = []

	for target in targets_names:
		train_images, gallery_images, queries_images = load_dataset(target)
		train_images_target.append(train_images)
		gallery_images_target.append(gallery_images)
		queries_images_target.append(queries_images)


	return train_images_target, gallery_images_target, queries_images_target

class sample(Dataset):
    
    def __init__(self, Set, segmentation_masks_dir, kind_of_transform=0):

        self.set = Set
        self.segmentation_masks_dir = segmentation_masks_dir
        self.kind_of_transform = kind_of_transform
        
        self.base_transform = Compose([Resize((256, 128), interpolation=functional.InterpolationMode.BICUBIC), ToTensor(), 
        	                                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.transform = Compose([Resize((256, 128), interpolation=functional.InterpolationMode.BICUBIC), 
                     RandomCrop((256, 128), padding=10), 
                     RandomHorizontalFlip(p=0.5), 
                     ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0), 
                     #RandomGrayscale(p=1.0),
                     ToTensor(), 
                     #RandomErasing(p=0.5, scale=(0.02, 0.04)),
                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.transform_clothes = Compose([Resize((256, 128), interpolation=functional.InterpolationMode.BICUBIC), 
                     #RandomCrop((256, 128), padding=10), 
                     RandomHorizontalFlip(p=0.5), 
                     ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
                     ToTensor(), 
                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.transform_vehicle = Compose([Resize((224, 224), interpolation=functional.InterpolationMode.BICUBIC), 
                     RandomCrop((224, 224), padding=10), 
                     RandomHorizontalFlip(p=0.5), 
                     ColorJitter(brightness=0.4, contrast=0.3, saturation=0.0, hue=0.0), 
                     ToTensor(),
                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

              
    def __getitem__(self, idx):
        
        sample = self.set[idx]
        imgPIL = torchreid.utils.tools.read_image(sample[0])
        reid_instance = sample[3]

        if reid_instance == "person":
            if self.kind_of_transform == 0:
                img_aug01 = torch.stack([self.transform(imgPIL)])
                img_aug02 = torch.stack([self.transform(imgPIL)])

            elif self.kind_of_transform == 1 or self.kind_of_transform == 2:

                mask_name = sample[0].split("/")[-1][:-4] + '.npy'
                full_mask_path = os.path.join(self.segmentation_masks_dir, mask_name)
                logits_result = np.load(full_mask_path)
                parsing_result = np.argmax(logits_result, axis=2)
                img_transformed = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(imgPIL)
                
                if self.kind_of_transform == 1:
                	mask_biometrics_traits = np.where((parsing_result == 0) | (parsing_result >= 13), 1, 0)
                elif self.kind_of_transform == 2:
                    mask_biometrics_traits = np.where(parsing_result >= 13, 1, 0)

                mask_no_biometrics_traits = 1 - mask_biometrics_traits

                img_tensor = ToTensor()(imgPIL)
                img_transformed_tensor = ToTensor()(img_transformed)

                img_aug01 = img_transformed_tensor*mask_no_biometrics_traits + img_tensor*mask_biometrics_traits
                imgPIL_aug01 = ToPILImage()(img_aug01)

                img_aug01 = torch.stack([self.base_transform(imgPIL)])
                img_aug02 = torch.stack([self.base_transform(imgPIL_aug01)])

        elif reid_instance == "object":
            img_aug01 = torch.stack([self.transform_vehicle(imgPIL)])
            img_aug02 = torch.stack([self.transform_vehicle(imgPIL)])

        return img_aug01, img_aug02
                 
    def __len__(self):
        return self.set.shape[0]

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

		self.msa01 = MultiheadAttention(256, 1, dropout=0.0, bias=False, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=True)
		self.msa02 = MultiheadAttention(512, 1, dropout=0.0, bias=False, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=True)
		self.msa03 = MultiheadAttention(1024, 1, dropout=0.0, bias=False, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=True)
		self.msa04 = MultiheadAttention(2048, 1, dropout=0.0, bias=False, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=True)

		self.global_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
		self.global_maxpool = AdaptiveMaxPool2d(output_size=(1, 1))
		self.last_bn = BatchNorm1d(2048)

	
	def forward(self, x, attention_map=False):
		
		x = self.conv1(x)
		x = self.bn1(x)
		#x = self.relu(x) # Do not discomment!
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x_featuremap = self.layer4(x)

		x = self.TransformerAttention(x_featuremap, "layer4")

		x_avg = self.global_avgpool(x)
		#x_max = self.global_maxpool(x_featuremap)
		x = x_avg #+ x_max
		#x = x_max
		output = x.view(x.size(0), -1)

		output = self.last_bn(output)

		#if not self.training:
		#	return output

		#output = self.fc01(output)
		#output = ReLU()(output)

		return output

	def TransformerAttention(self, x, layer_name):

		n, c, h, w = x.shape
		#print(n, c, h, w)

		# (n, dim, h, w) -> (n, dim, (h * w))
		x_r = x.reshape(n, c, h*w)
		
		# (n, dim, (h * w)) -> (n, (h * w), dim)
		# The self attention layer expects inputs in the format (N, S, E)
		# where S is the source sequence length, N is the batch size, E is the
		# embedding dimension
		x_rp = x_r.permute(0, 2, 1)
		
		if layer_name == "layer1":
			x, attn_x = self.msa01(x_rp, x_rp, x_rp)
		elif layer_name == "layer2":
			x, attn_x = self.msa02(x_rp, x_rp, x_rp)
		elif layer_name == "layer3":
			x, attn_x = self.msa03(x_rp, x_rp, x_rp)
		elif layer_name == "layer4":
			x, attn_x = self.msa04(x_rp, x_rp, x_rp)

		x = ReLU()(x)

		# Resize back to original dimension to return 
		x = x.permute(0, 2, 1)
		x = x.reshape(n, c, h, w)

		return x

class inceptionV3ReID(Module):
    
	def __init__(self, model_base):
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


def freezeLayers(model_name, model_online):

	if model_name == "resnet50":
		# Conv1
		model_online.module.conv1.training = False
		for i, param in enumerate(model_online.module.conv1.parameters()):
			assert param.requires_grad != None
			param.requires_grad = False

		# BN1
		model_online.module.bn1.training = False
		for i, param in enumerate(model_online.module.bn1.parameters()):
			assert param.requires_grad != None
			param.requires_grad = False

		# layer1
		model_online.module.layer1.training = False
		for i, param in enumerate(model_online.module.layer1.parameters()):
			assert param.requires_grad != None
			param.requires_grad = False

		# layer2
		model_online.module.layer2.training = False
		for i, param in enumerate(model_online.module.layer2.parameters()):
			assert param.requires_grad != None
			param.requires_grad = False

		# layer3
		model_online.module.layer3.training = False
		for i, param in enumerate(model_online.module.layer3.parameters()):
			assert param.requires_grad != None
			param.requires_grad = False	

	elif model_name == "osnet":

		# Conv1
		freezeLayer(model_online.module.conv1)

		# Conv2
		freezeLayer(model_online.module.conv2)

		# Conv3
		#freezeLayer(model_online.module.conv3)

	elif model_name == "densenet121":

		# Conv0
		freezeLayer(model_online.module.model_base.conv0)

		# Norm0
		freezeLayer(model_online.module.model_base.norm0)

		# denseblock1
		freezeLayer(model_online.module.model_base.denseblock1)
		freezeLayer(model_online.module.model_base.transition1)

		# denseblock2
		freezeLayer(model_online.module.model_base.denseblock2)
		freezeLayer(model_online.module.model_base.transition2)

		# denseblock3
		#freezeLayer(model_online.module.model_base.denseblock3)
		#freezeLayer(model_online.module.model_base.transition3)

	
	elif model_name == "efficientnetB0":

		no_freeze = True
		freezeLayer(model_online.module.features[0])
		freezeLayer(model_online.module.features[1])
		freezeLayer(model_online.module.features[2])
		freezeLayer(model_online.module.features[3])
		freezeLayer(model_online.module.features[4])
		freezeLayer(model_online.module.features[5])
		freezeLayer(model_online.module.features[6])
	

	elif model_name == "TransReID":

		no_freeze = True
		freezeLayer(model_online.module.base.patch_embed)
		freezeLayer(model_online.module.base.blocks[0])
		freezeLayer(model_online.module.base.blocks[1])
		freezeLayer(model_online.module.base.blocks[2])
		freezeLayer(model_online.module.base.blocks[3])
		freezeLayer(model_online.module.base.blocks[4])
		freezeLayer(model_online.module.base.blocks[5])
		freezeLayer(model_online.module.base.blocks[6])
		freezeLayer(model_online.module.base.blocks[7])
		freezeLayer(model_online.module.base.blocks[8])

	elif model_name == "ViT":

		print("Frozen!")
		freezeLayer(model_online.module.conv_proj)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_0)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_1)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_2)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_3)
		#NSA
		freezeLayer(model_online.module.encoder.layers.encoder_layer_4)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_5)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_6)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_7)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_8)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_9)
		freezeLayer(model_online.module.encoder.layers.encoder_layer_10)
		

	
			
def freezeLayer(layer):

	layer.training = False
	for i, param in enumerate(layer.parameters()):
		assert param.requires_grad != None
		param.requires_grad = False

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Define the UDA parameters')
	
	parser.add_argument('--gpu_ids', type=str, default="7", help='GPU IDs')
	parser.add_argument('--uccs_cluster', action="store_true", help="True if UCCS server is used, False it's assumed UNICAMPs is used")
	parser.add_argument('--model_name', type=str, help='Backbone name')
	parser.add_argument('--model_path', type=str, help='Path to the pre-trained weights')
	parser.add_argument('--multiview', type=str, help='If multi view must be used on training')
	parser.add_argument('--lr', type=float, default=3.5e-5, help='Learning Rate')
	parser.add_argument('--batch_size', type=int, default=128, help='Number of Persons')
	parser.add_argument('--lambda_bt', type=float, default=5e-3, help='Number of samples per person')
	parser.add_argument('--num_epoches', type=int, default=100, help='tau value used on softmax triplet loss')	
	parser.add_argument('--targets', type=str, help='Name of target dataset')
	parser.add_argument('--segmentation_masks_dir', type=str, help='Path to the segmentation masks if any')
	parser.add_argument('--kind_of_transform', type=int, default=0, 
												help='Kind of transform on augmentation: 0 (regular), (1) with background, (2) no background')
	parser.add_argument('--path_to_train_event', type=str, 
													help='Path to the directory with unlabeled images of an event (it must contain only people or only objects by now)')
	parser.add_argument('--path_to_save_models', type=str, help='Path to save models')
	parser.add_argument('--path_to_save_metrics', type=str, help='Path to save metrics (mAP, CMC, ...)')
	parser.add_argument('--version', type=str, help='Path to save models')
	parser.add_argument('--eval_freq', type=int, help='Evaluation Frequency along training')
	
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	uccs_cluster = args.uccs_cluster
	model_name = args.model_name
	model_path = args.model_path

	multiview = args.multiview
	if multiview == "True":
		multiview = True
	else:
		multiview = False

	base_lr = args.lr
	batch_size = args.batch_size
	lambda_bt = args.lambda_bt
	num_epoches = args.num_epoches
	targets = args.targets
	segmentation_masks_dir = args.segmentation_masks_dir
	kind_of_transform = args.kind_of_transform
	path_to_train_event = args.path_to_train_event
	dir_to_save = args.path_to_save_models
	dir_to_save_metrics = args.path_to_save_metrics
	version = args.version
	eval_freq = args.eval_freq


	# Nossa Senhora de Fátima
	main(gpu_ids, uccs_cluster, model_name, model_path, multiview, base_lr, batch_size, lambda_bt, num_epoches, targets, 
								segmentation_masks_dir, kind_of_transform, path_to_train_event, dir_to_save, dir_to_save_metrics, version, eval_freq)
