import os
import copy
import yaml

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
from validateModels import validateOnDatasets, calculateMetrics, validation
from datasetUtils import get_dataset_samples_and_statistics, load_dataset
from train_encoders import train

from random import shuffle
from termcolor import colored
from collections import defaultdict

from sklearn.metrics import pairwise_distances, silhouette_samples, normalized_mutual_info_score
from sklearn.cluster import DBSCAN, KMeans, OPTICS, cluster_optics_xi
from GPUClustering import GPUOPTICS, GPUDBSCAN, SelectClustersByQuality, selfAdaptiveGPUDBSCAN, EnsembleGPUDBSCAN, clusterFainessGPUDBSCAN
from GPUClustering import defineEpsForDBSCAN

from getFeatures import extractFeatures, get_subset_one_encoder
from torch.backends import cudnn

import faiss
from faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu
from tabulate import tabulate

import multiprocessing as mp
#from PEACH import PEACH

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

# Test evaluation with proxies!! MEasure distance between galleries and queries as references to the proxies!

def main(config):

	# Assigning parameter values from config file
	gpu_ids = config['gpu_ids']
	single_gpu_for_clustering = config['single_gpu_for_clustering']
	uccs_cluster = config['uccs_cluster'] 
	clustering_method = config['clustering_method'] 
	backbones_names = config['backbones']['names']
	backbones_weights = config['backbones']['weights']
	base_lr = config['train']['base_lr']
	P = config['train']['P']
	K = config['train']['K']
	tau = config['train']['tau']
	beta = config['train']['beta']
	k = config['train']['k']
	sampling = config['train']['sampling']  
	eps_value = config['train']['eps_value']
	eps_scheduling = config['train']['eps_scheduling']
	lambda_hard = config['train']['lambda_hard']
	number_of_epochs = config['train']['number_of_epochs']
	cotraining = config['train']['co_training']
	momentum_on_feature_extraction = config['train']['momentum_on_feature_extraction']
	percentage_from_whole_data = config['train']['percentage_from_whole_data']
	target = config['target']
	path_to_train_event = config['path_to_train_event']
	dir_to_save = config['path_to_save_models']  
	dir_to_save_metrics = config['path_to_save_metrics']
	version = config['version']
	eval_freq = config['eval_freq']

	########### DO NOT FORGET TO CITE THOSE GUYS FROM SYNC BN: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch!!!!! ##########################
	print("Git Branch:", os.system("git branch"))
	print(gpu_ids)
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

	num_gpus = torch.cuda.device_count()
	print("Num GPU's:", num_gpus)

	if single_gpu_for_clustering:
		if len(gpu_ids) > 1:
			gpu_indexes = np.arange(num_gpus)[1:].tolist()
		else:
			gpu_indexes = [0]

	else:
		gpu_indexes = np.arange(num_gpus).tolist()

	print("Allocated GPU's for model:", gpu_indexes)

	assert P % len(gpu_indexes) == 0, "The number of identities on batch (P) should be divisble by number of GPUs to train the model to avoid unbalancing"
	
	number_of_backbones = len(backbones_names)
	models = {}
	for backbone_idx, backbone_name in enumerate(backbones_names):
		models[backbone_name] = {}
		backbone_online, backbone_momentum = getDCNN(gpu_indexes, backbone_name, model_path=backbones_weights[backbone_idx])
		models[backbone_name]['online'] = backbone_online
		models[backbone_name]['momentum'] = backbone_momentum
		models[backbone_name]['optimizer'] = torch.optim.Adam(backbone_online.parameters(), lr=base_lr, weight_decay=5e-4)
	
	#print(R50_online)
	if target == "Places":
		train_images_target, evaluation_images_target = load_dataset(target, uccs_cluster)
		validation_engine = validation(target, None, None, evaluation_images=evaluation_images_target, gpu_index=gpu_indexes[0], verbose=0)
		#validation_engine.validateCheckpoint(Rmode)		

	elif "VehicleID" in target:
		train_images_target, evaluation_images_target = load_dataset(target, uccs_cluster)
		validation_engine = validation(target, None, None, evaluation_images=evaluation_images_target, gpu_index=gpu_indexes[0], verbose=0)

		distmats = []
		for backbone_name in backbones_names:
			model_distmat = validation_engine.validateCheckpoint(models[backbone_name]['online'])
			distmats.append(model_distmat)
		
		ensemble_distmats = np.mean(distmats, axis=0)
		validation_engine.calculateMetrics(ensemble_distmats)

	else:
		train_images_target, gallery_images_target, queries_images_target = get_dataset_samples_and_statistics([target], uccs_cluster)

		distmats = []
		for backbone_name in backbones_names:
			_, _, distmat = validateOnDatasets([target], queries_images_target, gallery_images_target, backbone_name, models[backbone_name]['online'], gpu_indexes)
			distmats.append(distmat)
		
		ensemble_distmats = np.mean(distmats, axis=0)
		calculateMetrics(ensemble_distmats, queries_images_target[0], gallery_images_target[0])


	print("Training dimension", train_images_target.shape)

	subset_images_target = train_images_target
	NT = train_images_target.shape[0]

	cmc_progress = []
	mAP_progress = []
	reliable_data_progress = []
	lr_values = []

	base_lr_values01 = np.linspace(base_lr/10, base_lr, num=10)

	if number_of_epochs > 10:
		base_lr_values02 = np.linspace(base_lr, base_lr, num=(number_of_epochs-10))
		base_lr_values = np.concatenate((base_lr_values01, base_lr_values02))
	else:
		base_lr_values = base_lr_values01

	
	# initializing times
	total_feature_extraction_time = 0
	total_clustering_time = 0
	total_finetuning_time = 0

	#number_of_iterations = 7
	minPts = 4
	
	t0_pipeline = time.time()

	#while True:
	for pipeline_iter in range(1, number_of_epochs+1):

		t0 = time.time()
		print("###============ Iteration number %d/%d ============###" % (pipeline_iter, number_of_epochs))
		
		if (pipeline_iter-1) % 3 == 0 and percentage_from_whole_data < 1.0:
			print(colored("Generating partition ...", "blue"))

			model_selected_for_feature_extraction = np.random.choice(backbones_names)
			selected_anchor_idx = np.random.choice(NT)

			print(colored("%s selected!" % model_selected_for_feature_extraction, "blue"))
			train_fvs = extractFeatures(train_images_target, models[model_selected_for_feature_extraction]['online'], 500, gpu_index=gpu_indexes[0])
			train_fvs = train_fvs/torch.norm(train_fvs, dim=1, keepdim=True)
			anchor_fv = train_fvs[selected_anchor_idx:(selected_anchor_idx+1)]
			anchor_similarities = torch.mm(anchor_fv, train_fvs.T)
			
			print(anchor_similarities.shape)
			subset_idx = torch.argsort(anchor_similarities[0], descending=True)[:int(NT*percentage_from_whole_data)]
			subset_images_target = train_images_target[subset_idx]
			
		print(colored("Number of images in current partition: %d" % subset_images_target.shape[0], "yellow"))

		if momentum_on_feature_extraction == False:
			print("Extracting Online Features ...")
			train_fvs = {}
			for backbone_name in backbones_names:
				fvs = extractFeatures(subset_images_target, models[backbone_name]['online'], 500, gpu_index=gpu_indexes[0])
				fvs = fvs/torch.norm(fvs, dim=1, keepdim=True)
				train_fvs[backbone_name] = fvs

		else:
			print("Extracting Ensembled Features ...")
			train_fvs = {}
			for backbone_name in backbones_names:
				fvs = extractFeatures(subset_images_target, models[backbone_name]['momentum'], 500, gpu_index=gpu_indexes[0])
				fvs = fvs/torch.norm(fvs, dim=1, keepdim=True)
				train_fvs[backbone_name] = fvs
			
		tf = time.time()
		dt_feature_extraction = tf - t0
		total_feature_extraction_time += dt_feature_extraction

		# Defining eps
		if eps_value:
			eps = eps_value
		
		elif eps_scheduling == "scheduling01":
			if pipeline_iter <= (3*number_of_epochs//4):
				if pipeline_iter <= number_of_epochs//2:
					eps = 0.7
				else:
					eps = getValueFromCosineSchedule(pipeline_iter, number_of_epochs//2, n_min=0.4, n_max=0.7)

		elif eps_scheduling == "scheduling02":
			if pipeline_iter <= number_of_epochs//2:
				eps = getValueFromCosineSchedule(pipeline_iter, number_of_epochs//2, n_min=0.4, n_max=0.7)

		elif eps_scheduling == "scheduling03":
			#if pipeline_iter <= (3*number_of_epochs//4):
			eps = getValueFromCosineSchedule(pipeline_iter, number_of_epochs, n_min=0.4, n_max=0.7)
		
		elif eps_scheduling == "scheduling04":
			#if pipeline_iter <= (3*number_of_epochs//4):
			eps = getValueFromCosineSchedule(pipeline_iter+number_of_epochs, number_of_epochs, n_min=0.4, n_max=0.7)

		elif eps_scheduling == "scheduling05":
				#if pipeline_iter <= (3*number_of_epochs//4):
				eps = getValueFromCosineSchedule(pipeline_iter, number_of_epochs//2, n_min=0.4, n_max=0.7)
				
		elif eps_scheduling == "scheduling06":
				#if pipeline_iter <= (3*number_of_epochs//4):
				eps = getValueFromCosineSchedule(pipeline_iter, number_of_epochs//3, n_min=0.4, n_max=0.7)

		elif eps_scheduling == "proposed":
			if pipeline_iter <= (3*number_of_epochs//4):
				eps = getValueFromCosineSchedule(pipeline_iter, number_of_epochs//2, n_min=0.4, n_max=0.7)

		t0_clustering = time.time()
		clustering_results = {}
		if clustering_method == "LRR": 
			for backbone_name in backbones_names:
				_, D, I = defineEpsForDBSCAN(train_fvs[backbone_name], k, minPts, pipeline_iter, 
												single_gpu_for_clustering=single_gpu_for_clustering, v="%s_%s" % (backbone_name, version))
				print(colored("Eps value for %s: %.2f" % (backbone_name, eps), "cyan"))
				pseudo_labels, core_points = GPUDBSCAN(D, I, k, minPts=minPts, eps=eps)
				non_outliers_idx = np.where(pseudo_labels != -1)[0]

				clustering_results[backbone_name] = {}
				clustering_results[backbone_name]['pseudo_labels'] = pseudo_labels
				clustering_results[backbone_name]['non_outlier_idx'] = non_outliers_idx

				ratio_of_reliable_data = subset_images_target[non_outliers_idx].shape[0]/subset_images_target.shape[0]
				print("Reliability after cluster selection: %.3f for %s" % (ratio_of_reliable_data, backbone_name))

				num_classes = len(np.unique(pseudo_labels[non_outliers_idx]))
				calculatePurityAndCameras(subset_images_target[non_outliers_idx], pseudo_labels[non_outliers_idx])
				print("Number of classes: %d" % num_classes)
	
		tf_clustering = time.time()
		dt_clustering = tf_clustering - t0_clustering
		total_clustering_time += dt_feature_extraction
		
		number_of_iterations = 5
		tf = time.time()
		
		# finetuning
		t0 = time.time()
		lr_value = base_lr_values[pipeline_iter-1]
		lr_values.append(lr_value)

		print(colored("Learning Rate: %f" % lr_value, "cyan"))
		for backbone_name in backbones_names:
			lambda_lr_warmup(models[backbone_name]['optimizer'], lr_value)
		

		if cotraining:
			
			selected_backbones = {}
			switch = False
			for backbone_name in backbones_names:
				not_candidates = list(selected_backbones.values()) + [backbone_name]
				candidates = np.setdiff1d(backbones_names, not_candidates)
				if len(candidates) > 0:
					selected = np.random.choice(candidates, replace=False)
					selected_backbones[backbone_name] = selected
				else:
					switch = True

			if switch:
				selected_for_the_last = np.random.choice(backbones_names[:-1], replace=False)
				selected_backbones[backbones_names[-1]] = selected_backbones[selected_for_the_last]
				selected_backbones[selected_for_the_last] = backbones_names[-1]

			assert sum([k == v for k,v in selected_backbones.items()]) == 0
			assert sum(np.unique(list(selected_backbones.values()), return_counts=True)[1]) == number_of_backbones
			assert sum(np.unique(list(selected_backbones.keys()), return_counts=True)[1]) == number_of_backbones
		else:
			pseudo_labels_idx = np.arange(3)

		#pseudo_labels_idx = np.random.choice([0,1,2], size=3, replace=False)
		print('Co-Training:')
		print(selected_backbones)

		for backbone_name in backbones_names:
			print(colored("Training %s with pseudo-labels from %s with K = %d ..." % (backbone_name, selected_backbones[backbone_name], K), "green"))
			selected_pseudo_labels = clustering_results[selected_backbones[backbone_name]]['pseudo_labels']
			selected_outliers_idx  = clustering_results[selected_backbones[backbone_name]]['non_outlier_idx'] 
			model_online = models[backbone_name]['online']
			model_momentum = models[backbone_name]['momentum']
			optimizer = models[backbone_name]['optimizer']
			updated_model_online, updated_model_momentum, updated_optimizer = train(subset_images_target, selected_pseudo_labels, None, selected_outliers_idx, 
																						sampling, optimizer, P, K, tau, beta, lambda_hard, number_of_iterations, 
																																model_online, model_momentum, gpu_indexes)

			models[backbone_name]['online'] = updated_model_online
			models[backbone_name]['momentum'] = updated_model_momentum
			models[backbone_name]['optimizer'] = updated_optimizer

			torch.save(updated_model_online.state_dict(), "%s/model_online_%s_%s.h5" % (dir_to_save, backbone_name, version))
			torch.save(updated_model_momentum.state_dict(), "%s/model_momentum_%s_%s.h5" % (dir_to_save, backbone_name, version))
		
		tf = time.time()
		dt_finetuning = tf - t0
		total_finetuning_time += dt_finetuning
		
		if pipeline_iter % eval_freq == 0:

			if "VehicleID" in target:
				train_images_target, evaluation_images_target = load_dataset(target, uccs_cluster)
				validation_engine = validation(target, None, None, evaluation_images=evaluation_images_target, gpu_index=gpu_indexes[0], verbose=0)

				distmats = []
				for backbone_name in backbones_names:
					model_distmat = validation_engine.validateCheckpoint(models[backbone_name]['online'])
					distmats.append(model_distmat)
				
				ensemble_distmats = np.mean(distmats, axis=0)
				mAP, cmc = validation_engine.calculateMetrics(ensemble_distmats)

			else:
				distmats = []
				for backbone_name in backbones_names:
					validateOnDatasets([target], queries_images_target, gallery_images_target, "%s_online" % backbone_name, 
																				models[backbone_name]['online'], gpu_indexes)

					_, _, distmat = validateOnDatasets([target], queries_images_target, gallery_images_target, "%s_momentum" % backbone_name, 
																				models[backbone_name]['momentum'], gpu_indexes)
					
					distmats.append(distmat)

				ensemble_distmats = np.mean(distmats, axis=0)

				if "ImageNet" in [target]:
					cmc, mAP = calculateMetrics(ensemble_distmats, queries_images_target[0][-1], gallery_images_target[0][-1])
				else:
					cmc, mAP = calculateMetrics(ensemble_distmats, queries_images_target[0], gallery_images_target[0])
			
			cmc_progress.append(cmc)
			mAP_progress.append(mAP)

			joblib.dump(cmc_progress, "%s/CMC_%s" % (dir_to_save_metrics, version))
			joblib.dump(mAP_progress, "%s/mAP_%s" % (dir_to_save_metrics, version))

		
	tf_pipeline = time.time()
	total_pipeline_time = tf_pipeline - t0_pipeline

	number_of_epochs = pipeline_iter
	mean_feature_extraction_time = total_feature_extraction_time/number_of_epochs
	mean_clustering_time = total_clustering_time/number_of_epochs
	mean_finetuning_time = total_finetuning_time/number_of_epochs

	print(total_feature_extraction_time, total_clustering_time, total_finetuning_time)
	print("Mean Feature Extraction and Reranking Time: %f" % mean_feature_extraction_time)
	print("Mean Clustering Time: %f" % mean_clustering_time)
	print("Mean Finetuning Time: %f" % mean_finetuning_time)
	print("Total pipeline Time:  %f" % total_pipeline_time)





def outlierCorrection(model02_label, model03_label, model01_pseudo_labels, model02_pseudo_labels, model03_pseudo_labels):

	model02_cluster_indexes = np.where(model02_pseudo_labels == model02_label)[0]
	model03_cluster_indexes = np.where(model03_pseudo_labels == model03_label)[0]
	
	model02_to_model01_labels, model02_to_model01_votes = np.unique(model01_pseudo_labels[model02_cluster_indexes], return_counts=True) 
	model03_to_model01_labels, model03_to_model01_votes = np.unique(model01_pseudo_labels[model03_cluster_indexes], return_counts=True) 
	
	union_labels = np.union1d(model02_to_model01_labels, model03_to_model01_labels)
	total_votes = np.zeros(len(union_labels))

	for label_idx in np.arange(len(union_labels)):
		idx1 = np.where(union_labels[label_idx] == model02_to_model01_labels)[0]
		idx2 = np.where(union_labels[label_idx] == model03_to_model01_labels)[0]
		
		if len(idx1) == 0:
			total_votes[label_idx] = model03_to_model01_votes[idx2[0]]
		elif len(idx2) == 0:
			total_votes[label_idx] = model02_to_model01_votes[idx1[0]]
		else:
			total_votes[label_idx] = model03_to_model01_votes[idx2[0]] + model02_to_model01_votes[idx1[0]]

	if -1 in union_labels:
		j = 1
	else:
		j = 0

	# Checking if ALL samples from other models have been mapped to outliers
	if j == 1 and len(union_labels) == 1:
		return -1

	most_voted_idx = np.argmax(total_votes[j:])
	most_voted_cluster = union_labels[j:][most_voted_idx]
	return most_voted_cluster

#NSA
def selectProxiesByTriagulation(X, num_proxies=5):

    mean_fv = torch.mean(X, dim=0, keepdim=True)
    first_idx = torch.argmax(torch.cdist(mean_fv,X, p=2.0)).item()
    #NSA
    dist = torch.cdist(X,X,p=2.0)
    n = dist.shape[0]
    #cumulative_vector = torch.zeros(n)
    cumulative_vector = torch.ones(n)*torch.max(dist)
    proxies = [first_idx]

    num_proxies = min(num_proxies, n)

    i = 0
    #while True:
    for j in range(num_proxies-1):
        #previous_proxies = np.unique(proxies)
        sample_idx = proxies[i]
        #print(sample_idx)
        #cumulative_vector += dist[sample_idx]
        cumulative_vector = np.minimum(cumulative_vector, dist[sample_idx])
        #furthest_idx = torch.argmax(cumulative_vector)
        furthest_idx = torch.argsort(cumulative_vector)[-1]
        proxies.append(furthest_idx.item())
        #post_proxies = np.unique(proxies)

        #if len(previous_proxies) == len(post_proxies):
        #  break
        
        i += 1

    proxies = torch.Tensor(proxies).long()
    #proxies_fvs = X[proxies]
    max_dist_between_proxies = torch.max(dist[proxies, proxies]).item()
    #print(proxies)
    proxy_labels = torch.argmin(dist[:, proxies], dim=1) 
    #print(proxy_labels)
    
    return proxies, proxy_labels, max_dist_between_proxies

def getValueFromCosineSchedule(t_cur, t_max, n_min=0.0, n_max=1.0):
	nt = n_min + 0.5*(n_max - n_min)*(1 + np.cos(((t_max - t_cur)/t_max)*np.pi))
	return nt

def getImagesPseudoLabelsIterations(subset_images_target, pseudo_labels, num_times_to_check_all_samples, K):

	selected_images = subset_images_target[pseudo_labels != -1]
	pseudo_labels = pseudo_labels[pseudo_labels != -1]
	ratio_of_reliable_data = selected_images.shape[0]/subset_images_target.shape[0]
	print(colored("Reliability after cluster selection: %.3f" % ratio_of_reliable_data, "yellow"))
	calculatePurityAndCameras(selected_images, pseudo_labels)

	clusters, cluster_freqs = np.unique(pseudo_labels, return_counts=True)	
	number_of_iterations = num_times_to_check_all_samples*int(np.ceil(np.mean(cluster_freqs)/K))
	print("Considering mean: %d, based on num times to check all of samples: %d" % (number_of_iterations, num_times_to_check_all_samples))

	return selected_images, pseudo_labels, number_of_iterations


	
def getDatasetPartitions(images, perc_closest, model_online, gpu_indexes):

	partitions = []
	partition_size = int(int(len(images)*perc_closest))
	remaining_images = images 

	while len(remaining_images) >= partition_size:
		selected_idx = np.random.choice(len(remaining_images), replace=False)
		selected_sample = remaining_images[selected_idx]
		print(selected_idx, selected_sample)

		selected_indexes, non_selected_indexes  = get_subset_one_encoder(selected_sample, remaining_images, 
																			partition_size, model_online, batch_size=500, 
																											gpu_index=gpu_indexes[0])

		partitions.append(remaining_images[selected_indexes])
		remaining_images = remaining_images[non_selected_indexes]


	for pidx in range(len(partitions)):
		print("Partition %d has %d samples" % (pidx, len(partitions[pidx])))

	return partitions


def calculatePurityAndCameras(images, pseudo_labels):

	unique_pseudo_labels = np.unique(pseudo_labels)

	if unique_pseudo_labels[0] == -1:
		j = 1
	else:
		j = 0

	H = 0
	cameras_quantities = []

	for lab in unique_pseudo_labels[j:]:
		#print(colored("Statistics on Cluster %d" % lab, "green"))
		cluster_images = images[pseudo_labels == lab]
		#print("Cluster Images:", cluster_images)
		cluster_true_ids = cluster_images[:,1]
		true_ids, freq_ids = np.unique(cluster_true_ids, return_counts=True)
		#print("ID's and Frequencies:", true_ids, freq_ids)
		ratio_freq_ids = freq_ids/np.sum(freq_ids)
		#print("Frequencies in ratio:", ratio_freq_ids)
		cluster_purity = np.sum(-1*ratio_freq_ids*np.log2(ratio_freq_ids))
		#print("Cluster Purity", cluster_purity)
		H += cluster_purity

		cluster_cameras, cameras_freqs = np.unique(cluster_images[:,2], return_counts=True)
		#print("Cameras and Frequencies:", cluster_cameras, cameras_freqs)
		#print("There are %d cameras" % len(cameras_freqs))
		cameras_quantities.append(len(cameras_freqs))


	#print("Total number of clusters:", len(unique_pseudo_labels[j:]))
	mean_purity = H/len(unique_pseudo_labels[j:])
	NMI = normalized_mutual_info_score(images[:,1], pseudo_labels)
	print(colored("Mean Purity: %.5f" % mean_purity, "green"))
	print(colored("NMI: %.3f" % NMI, "green"))

	numbers_of_cameras, cameras_freqs = np.unique(cameras_quantities, return_counts=True)

	#for i in range(len(numbers_of_cameras)):
	#	print(colored("There are %d clusters with %d cameras" % (cameras_freqs[i], numbers_of_cameras[i]), "blue"))

def selectionBySparseness(selection_threshold, images, pseudo_labels, model, gpu_index=0):


	fvs = extractFeatures(images, model, 500, gpu_index=gpu_index)
	fvs = fvs/torch.norm(fvs, dim=1, keepdim=True)

	unique_pseudo_labels = np.unique(pseudo_labels)

	if unique_pseudo_labels[0] == -1:
		j = 1
	else:
		j = 0


	# Computer Silhouette
	silhouette_scores = silhouette_samples(fvs, pseudo_labels, metric='euclidean')

	# This data structure will hold the summation of sparseness along with the number of clusters per number of cameras
	# E.g., clusters with 2 cameras will have their sparseness summed and divided by the total number of clusters with two cameras
	sparseness_per_number_of_cameras = defaultdict()
	all_clusters_sparseness = []

	for lab in unique_pseudo_labels[j:]:
		#print(colored("Statistics on Cluster %d" % lab, "green"))
		cluster_images = images[pseudo_labels == lab]
		cluster_fvs = fvs[pseudo_labels == lab]
		cluster_silhouette = silhouette_scores[pseudo_labels == lab].mean()
		mean_cluster_fvs = torch.mean(cluster_fvs, dim=0, keepdims=True)
		mean_cluster_fvs = mean_cluster_fvs/torch.norm(mean_cluster_fvs, dim=1, keepdim=True)
		cluster_sparseness = torch.cdist(mean_cluster_fvs, cluster_fvs).mean()
		all_clusters_sparseness.append([lab, cluster_sparseness])

		freqs_of_true_classes = np.unique(cluster_images[:,1], return_counts=True)[1]
		max_occurence = np.max(freqs_of_true_classes)
		ACC = max_occurence/np.sum(freqs_of_true_classes)

		quality_score = (1.0 - (cluster_sparseness/2.0))*((cluster_silhouette+1)/2.0)

		print("Label: %d, Size: %d, Sparseness: %.5f, ACC: %.2f, Silhouette: %.2f, Quality: %.2f" % (lab, cluster_images.shape[0], 
																										cluster_sparseness, ACC, 
																										cluster_silhouette, quality_score))

		if quality_score >= 0.35:
			print(colored("Good Cluster!", "green"))
		else:
			print(colored("Bad Cluster!", "red"))

		cluster_cameras, cameras_freqs = np.unique(cluster_images[:,2], return_counts=True)
		number_of_cameras_on_cluster = len(cameras_freqs)
		
		if number_of_cameras_on_cluster in sparseness_per_number_of_cameras.keys():
			element = sparseness_per_number_of_cameras[number_of_cameras_on_cluster]
			element[0] += cluster_sparseness
			element[1] += 1
			sparseness_per_number_of_cameras[number_of_cameras_on_cluster] = element
		else:
			sparseness_per_number_of_cameras[number_of_cameras_on_cluster] = [cluster_sparseness, 1]

	sorted_keys = sorted(sparseness_per_number_of_cameras.keys())

	for key in sorted_keys:
		element = sparseness_per_number_of_cameras[key]
		mean_sparseness = element[0]/element[1]
		print(colored("The mean sparseness for clusters with %d cameras is: %f" % (key, mean_sparseness), "white"))


	threshold_position = int(len(all_clusters_sparseness)*selection_threshold)
	all_clusters_sparseness = np.array(all_clusters_sparseness)
	lower_bound_sparseness = np.sort(all_clusters_sparseness[:,1])[threshold_position]
	#mean_clusters_sparseness = np.mean(all_clusters_sparseness[:,1])
	#std_clusters_sparseness = np.std(all_clusters_sparseness[:,1])

	# Usually the sparseness does not have a gaussian distribution! So we remove 16% (from gaussian analysis)
	# of the lowerst clusters 
	#lower_bound_sparseness = mean_sparseness - std_clusters_sparseness
	print("Clusters with sparseness lower than %f will be removed" % lower_bound_sparseness)

	for cluster_idx in range(len(all_clusters_sparseness)):
		if all_clusters_sparseness[cluster_idx][1] < lower_bound_sparseness:
			clusted_id = all_clusters_sparseness[cluster_idx][0]
			pseudo_labels[pseudo_labels == clusted_id] = -1

	selected_images = images[pseudo_labels != -1]
	proposed_labels = pseudo_labels[pseudo_labels != -1]

	return selected_images, proposed_labels


def lambda_lr_warmup(optimizer, lr_value):

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr_value

def load_multiple_datasets(target_names):

	# Note that we will concatenate all training images but NOT the gallery and query images,
	# since we would like to have an unique training set but separated evaluations

	train_images_target = []
	gallery_images_target = []
	queries_images_target = []

	for target in target_names:

		#print("Dataset %s has:" % target)
		train_images, gallery_images, queries_images = load_dataset(target)
		#print(train_images.shape)
		#print(gallery_images.shape)
		#print(queries_images.shape)

		train_images_target.append(train_images)
		gallery_images_target.append(gallery_images)
		#NSA
		queries_images_target.append(queries_images)

	#train_images_target = np.concatenate(train_images_target, axis=0)
	#NSA - #print("Total train data size:", train_images_target.shape)

	return train_images_target, gallery_images_target, queries_images_target

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Define the parameters')
	
	parser.add_argument('--config_params_file_path', type=str, default="config.yaml", help='Set the path to the configuration parameters file')
	parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
	
	args = parser.parse_args()

	config_params_file_path = args.config_params_file_path

	# Loading yaml with the configuration parameters
	config_params = list(yaml.safe_load_all(open(config_params_file_path, "r")))[0]
	main(config_params)

	print("Santa Maria Mae de Deus")



