
import torch
import torchreid
import torchvision

from torchvision.models import resnet50, densenet121, inception_v3
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, RandomErasing, RandomHorizontalFlip, ColorJitter
from torchvision.transforms import RandomCrop, functional, GaussianBlur
from torch.nn import Module, Dropout, BatchNorm1d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, ReLU, AvgPool2d, AdaptiveMaxPool2d
from torch.nn import functional as F
from torch import nn

from torch.utils.data.distributed import DistributedSampler

import os
import copy
 
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import numpy as np
import time
import argparse
import joblib

from Encoders import getDCNN, getEnsembles
from validateModels import validateOnDatasets
from datasetUtils import get_dataset_samples_and_statistics
from losses import BatchCenterLoss, BatchSoftmaxAllTripletLoss, BatchSoftmaxTripletLoss

from random import shuffle
from termcolor import colored

from sklearn.cluster import DBSCAN, KMeans

from getFeatures import extractFeatures
from torch.backends import cudnn

import faiss
from faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu


transform_person_augmentation = Compose([Resize((256, 128), interpolation=functional.InterpolationMode.BICUBIC), 
                     RandomCrop((256, 128), padding=10), 
                     RandomHorizontalFlip(p=0.5), 
                     ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.0), 
                     ToTensor(), 
                     RandomErasing(p=1.0, scale=(0.02, 0.33)), 
                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_vehicle_augmentation = Compose([Resize((224, 224), interpolation=functional.InterpolationMode.BICUBIC), 
                     RandomCrop((224, 224), padding=10), 
                     RandomHorizontalFlip(p=0.5), 
                     ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.0), 
                     ToTensor(),
                     RandomErasing(p=1.0, scale=(0.02, 0.33)),
                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

#def train(queue, key_name, selected_images, pseudo_labels, proxies_pseudo_labels, non_outliers_idx, sampling, optimizer, P, K, tau, beta, lambda_hard, number_of_iterations, 
#																						model_online, model_momentum, gpu_indexes):
def train(selected_images, pseudo_labels, proxies_pseudo_labels, non_outliers_idx, sampling, optimizer, P, K, tau, beta, lambda_hard, number_of_iterations, 
																														model_online, model_momentum, gpu_indexes):
	
		centers_labels = np.unique(pseudo_labels[non_outliers_idx])
		num_classes = len(centers_labels)

		event_dataset = samplePKBatches(selected_images, pseudo_labels, non_outliers_idx, K=K, select_outliers=False)
		#event_dataset = samplePKBatchesSubRegions(selected_images, pseudo_labels, proxies_pseudo_labels, non_outliers_idx, K=K) 	
		#event_dataset.__getitem__(0)
		#exit()

		# For DDP
		#batchLoader = DataLoader(event_dataset, batch_size=min(P, num_classes), num_workers=8, 
		#											pin_memory=True, shuffle=False, drop_last=True, collate_fn=collate_fn_PK, sampler=DistributedSampler(event_dataset))

		batchLoader = DataLoader(event_dataset, batch_size=min(P, num_classes), num_workers=8, pin_memory=True, shuffle=False, drop_last=True, collate_fn=collate_fn_PK)

		keys = list(model_online.state_dict().keys())
		size = len(keys) 

		#model_online.train()
		model_online.train()
		#freezeLayers(key_name, model_online)

		model_momentum.eval()

		num_batches_computed = 0
		lambda_redundancy = 5e-3
		lambda_instance = 1.0
		lambda_bt = 1e-3

		scaler = torch.cuda.amp.GradScaler(enabled=True)

		for inner_iter in np.arange(number_of_iterations):
		#while reach_max_number_iterations == False:

			print(colored("Iteration number: %d/%d" % (inner_iter+1, number_of_iterations), "green"))

			model_online.eval()


			if sampling == "mean" or sampling == "both":

				selected_feature_vectors = extractFeatures(selected_images[non_outliers_idx], model_online, 500, gpu_index=gpu_indexes[0])

				centers = []
				for label in centers_labels:

					if sampling == "mean":
						center = torch.mean(selected_feature_vectors[pseudo_labels[non_outliers_idx] == label], dim=0, keepdim=True)
					elif sampling == "both":
						center_mean = torch.mean(selected_feature_vectors[pseudo_labels[non_outliers_idx] == label], dim=0, keepdim=True)
						N_samples = np.sum(pseudo_labels[non_outliers_idx] == label)
						selected_sample_idx = np.random.choice(N_samples) 
						center_random = selected_feature_vectors[pseudo_labels[non_outliers_idx] == label][selected_sample_idx:(selected_sample_idx+1)]
						center = (center_mean + center_random)/2

					if len(centers) == 0:
						centers = center
					else:
						centers = torch.cat((centers, center), dim=0)

				centers = centers/torch.norm(centers, dim=1, keepdim=True)
				centers = centers.cuda(gpu_indexes[0])

			elif sampling == "random":

				centers_images = []
				for label in centers_labels:
					N_samples = np.sum(pseudo_labels[non_outliers_idx] == label)
					selected_sample_idx = np.random.choice(N_samples) 
					center_img = selected_images[non_outliers_idx][pseudo_labels[non_outliers_idx] == label][selected_sample_idx]
					centers_images.append(center_img)

				centers_images = np.array(centers_images)
				centers = extractFeatures(centers_images, model_online, 500, gpu_index=gpu_indexes[0])
				centers = centers/torch.norm(centers, dim=1, keepdim=True)
				centers = centers.cuda(gpu_indexes[0])



			#torch.save(centers, 'centers_Duke.pt')
			#exit()

			#model_online.train()
			model_online.train()
			#freezeLayers(key_name, model_online)


			iteration_loss = 0.0
			iteration_center = 0.0
			iteration_hard = 0.0
			iteration_instance = 0.0

			total_corrects = 0
			total_batch_size = 0

			for batch_idx, batch in enumerate(batchLoader):

				initilized = False
				for imgs, labels, pids in batch:

					if initilized:
						batch_imgs = torch.cat((batch_imgs, imgs), dim=0)
						#batch_labels = torch.cat((batch_labels, labels), dim=0)
						batch_labels = np.concatenate((batch_labels,labels), axis=0)
						batch_pids = np.concatenate((batch_pids, pids), axis=0)
					else:
						batch_imgs = imgs
						batch_labels = labels
						batch_pids = pids
						initilized = True

				batch_imgs = batch_imgs.cuda(gpu_indexes[0])

				if batch_imgs.shape[0] <= 2:
					continue

				# Feature Vectors of Original Images
				#print(batch_imgs.shape)
				with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):

					batch_fvs = model_online(batch_imgs)
					batch_fvs = batch_fvs/torch.norm(batch_fvs, dim=1, keepdim=True)

					#print(batch_fvs.shape)

					batch_center_loss = BatchCenterLoss(batch_fvs, batch_labels, centers, centers_labels, tau=tau, 
																										gpu_index=gpu_indexes[0])

					#batch_hard_loss = BatchSoftmaxAllTripletLoss(batch_fvs, batch_labels, batch_pids, tau=tau, gpu_index=gpu_indexes[0])

					batch_hard_loss, corrects, total_number_triplets = BatchSoftmaxTripletLoss(batch_fvs, batch_labels, 
																											batch_pids, tau=tau, 
																											gpu_index=gpu_indexes[0])
					#batch_center_loss = BatchCenterLossWithOutliers(batch_fvs, batch_labels, centers, centers_labels, tau=tau, 
					#																					gpu_index=gpu_indexes[0])
					#batch_hard_loss, corrects, total_number_triplets = BatchSoftmaxTripletLossWithOutliers(batch_fvs, batch_labels, 
					#																						batch_pids, tau=tau, 
					#																						gpu_index=gpu_indexes[0])

					#corrects = 0.0
					#total_number_triplets = 1

					#batch_hard_loss = BatchWeightedSoftmaxAllTripletLoss(batch_fvs, batch_labels, gpu_index=gpu_indexes[0])

					#batch_instance_loss = BatchInstanceLossOutliers(batch_fvs, batch_labels, batch_pids, gpu_index=0)
					#batch_hard_loss, corrects, total_number_triplets = BatchMedianSoftmaxTripletLoss(batch_fvs, batch_labels, 
					#																						batch_pids, tau=tau, 
					#																						gpu_index=gpu_indexes[0])

					batch_loss = batch_center_loss + lambda_hard*batch_hard_loss #+ lambda_instance*batch_instance_loss
				
				total_corrects += corrects
				total_batch_size += total_number_triplets

				iteration_center += batch_center_loss.item()
				iteration_hard += batch_hard_loss.item()
				#iteration_instance += batch_instance_loss.item()
				iteration_loss += batch_loss.item()

				optimizer.zero_grad()
				scaler.scale(batch_loss).backward()
				#batch_loss.backward()
				scaler.step(optimizer)
				#optimizer.step()
				scaler.update()

				#centers = UpdateCenters(batch_fvs, batch_labels, centers, centers_labels, alpha=alpha)
				model_online.eval()
				model_online_weights = model_online.state_dict()
				model_momentum_weights = model_momentum.state_dict()

				for i in range(size):	
					model_momentum_weights[keys[i]] =  beta*model_momentum_weights[keys[i]] + (1-beta)*model_online_weights[keys[i]].detach()
					
				model_momentum.load_state_dict(model_momentum_weights)
				#model_online.train()
				model_online.train()
				#freezeLayers(key_name, model_online)


				num_batches_computed += 1
				#if total_number_of_batches >= number_of_iterations:
				#	reach_max_number_iterations = True
				#	break

			TTPR = total_corrects/total_batch_size

			iteration_loss = iteration_loss/(batch_idx+1)
			iteration_center = iteration_center/(batch_idx+1)
			iteration_hard = iteration_hard/(batch_idx+1)
			#iteration_instance = iteration_instance/(batch_idx+1)
	
			print(colored("Batches computed: %d, Tau value: %.3f" % (num_batches_computed, tau), "cyan"))
			print(colored("Mean Loss: %.7f, Mean Center Loss: %.7f, Mean Hard Triplet Loss: %.7f" % (iteration_loss, 
																									iteration_center, 
																									iteration_hard), "yellow"))
			
			
			print(colored("Percetage of correct triplets: %.2f" % TTPR, "blue"))
			#progress_loss.append(iteration_loss)
			
		model_online.eval()
		model_momentum.eval()

		#torch.save(model_online.state_dict(), "temp_model_online_%s.h5" % key_name)
		#torch.save(model_momentum.state_dict(), "temp_model_momentum_%s.h5" % key_name)
		#torch.save(optimizer.state_dict(), "temp_optimizer_%s.h5" % key_name)

		#queue.put({key_name: ["temp_model_online_%s.h5" % key_name, "temp_model_momentum_%s.h5" % key_name, "temp_optimizer_%s.h5" % key_name]})
		return model_online, model_momentum, optimizer
		

def freezeLayers(model_name, model_online):

	if model_name == "R50":
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
		#model_online.module.layer3.training = False
		#for i, param in enumerate(model_online.module.layer3.parameters()):
		#	assert param.requires_grad != None
		#	param.requires_grad = False	

	elif model_name == "OSN":

		# Conv1
		freezeLayer(model_online.module.conv1)

		# Conv2
		freezeLayer(model_online.module.conv2)

		# Conv3
		#freezeLayer(model_online.module.conv3)

	elif model_name == "DEN":

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
	


def collate_fn(batch):
	return torch.cat(batch, dim=0)

def collate_fn_PK(batch):
	return batch

class samplePKBatches(Dataset):
    
	def __init__(self, images, pseudo_labels, non_outliers_idx, K=4, perc=1.0, select_outliers=False):

		self.images_names = images[:,0]
		self.true_ids = images[:,1]
		self.reid_instances = images[:,3]
		self.pseudo_labels = pseudo_labels
		self.labels_set = np.unique(pseudo_labels[non_outliers_idx])
		self.K = K
		self.select_outliers = select_outliers

		np.random.shuffle(self.labels_set)
		self.num_of_pseudo_identities_on_epoch = int(round(len(self.labels_set)*perc))
		
	def __getitem__(self, idx):

		pseudo_identity = self.labels_set[idx]
		images_identity = self.images_names[self.pseudo_labels == pseudo_identity]
		true_identities = self.true_ids[self.pseudo_labels == pseudo_identity]
		reid_instances = self.reid_instances[self.pseudo_labels == pseudo_identity]
		
		if images_identity.shape[0] >= self.K:
			selected_images_idx = np.random.choice(images_identity.shape[0], size=self.K, replace=False)
		else:
			selected_images_idx = np.random.choice(images_identity.shape[0], size=self.K, replace=True)


		#selected_images_idx = np.random.choice(images_identity.shape[0], size=min(images_identity.shape[0], self.K), replace=False)
	
		selected_images = images_identity[selected_images_idx]
		selected_true_identities = true_identities[selected_images_idx]
		selected_reid_instances = reid_instances[selected_images_idx]
		
		batch_images = []
		for img_idx in np.arange(len(selected_images)):

			img_name = selected_images[img_idx] 
			imgPIL = torchreid.utils.tools.read_image(img_name)
			reid_inst = selected_reid_instances[img_idx]

			if reid_inst == 'person':
				augmented_img = torch.stack([transform_person_augmentation(imgPIL)])
			elif reid_inst == 'object' or reid_inst == 'imagenet' or reid_inst == 'place':
				augmented_img = torch.stack([transform_vehicle_augmentation(imgPIL)])	

			if len(batch_images) == 0:
				batch_images = augmented_img
			else:
				batch_images = torch.cat((batch_images, augmented_img), dim=0)

		batch_labels = np.array(batch_images.shape[0]*[pseudo_identity])

		if self.select_outliers:

			images_identity = self.images_names[self.pseudo_labels == '-1.0_-1.0_-1.0']
			true_identities = self.true_ids[self.pseudo_labels == '-1.0_-1.0_-1.0']
			reid_instances = self.reid_instances[self.pseudo_labels == '-1.0_-1.0_-1.0']

			# Verify if it indeed has outliers
			if len(images_identity) > 0:
				selected_images_idx = np.random.choice(images_identity.shape[0], replace=False)
				
				selected_images = images_identity[selected_images_idx]
				selected_true_identities = np.append(selected_true_identities, [true_identities[selected_images_idx], true_identities[selected_images_idx]])
				selected_reid_instances = reid_instances[selected_images_idx]
				
				imgPIL = torchreid.utils.tools.read_image(selected_images)
				reid_inst = selected_reid_instances

				if reid_inst == 'person':
					augmented_img01 = torch.stack([transform_person_augmentation(imgPIL)])
					augmented_img02 = torch.stack([transform_person_augmentation(imgPIL)])
				elif reid_inst == 'object' or reid_inst == 'imagenet' or reid_inst == 'place':
					augmented_img = torch.stack([transform_vehicle_augmentation(imgPIL)])	

				if len(batch_images) == 0:
					batch_images = torch.cat((augmented_img01, augmented_img02), dim=0)
				else:
					batch_images = torch.cat((batch_images, augmented_img01, augmented_img02), dim=0)


			batch_labels = np.append(batch_labels, ['-1.0_-1.0_-1.0','-1.0_-1.0_-1.0'])

		return batch_images, batch_labels, selected_true_identities
	
	def __len__(self):
		return self.num_of_pseudo_identities_on_epoch
