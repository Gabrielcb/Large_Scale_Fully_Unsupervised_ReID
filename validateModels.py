import torch
import torchreid
import torchvision
from torchvision.models import resnet50, densenet121, inception_v3 #, vit_b_16
from torch.nn import Module, Dropout, BatchNorm1d, BatchNorm2d, Linear, AdaptiveAvgPool2d, CrossEntropyLoss, Softmax, ReLU, AdaptiveMaxPool2d, Conv2d
from torch.nn import functional as F
from torch import nn

import numpy as np
from termcolor import colored
from tqdm import tqdm

from getFeatures import extractFeatures, extractFeaturesMultiView, extractFeaturesDual, extractFeaturesMultiPart
from sklearn.model_selection import KFold

import warnings

np.random.seed(12)

try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )

def validate(queries, gallery, model, rerank=False, gpu_index=0, verbose=1, sobel=False, grayscale=False):
    model.eval()
    queries_fvs = extractFeatures(queries, model, 500, gpu_index, sobel=sobel, grayscale=grayscale)
    gallery_fvs = extractFeatures(gallery, model, 500, gpu_index, sobel=sobel, grayscale=grayscale)

    queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
    gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

    distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
    distmat = distmat.numpy()

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
        distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
        distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)

    del queries_fvs, gallery_fvs
    # Compute Ranks
    ranks=[1, 5, 10, 20]
    if verbose >= 1:
        print('Computing CMC and mAP ...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)

    #del distmat
    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
        
    return cmc[:20], mAP, distmat


def validate_with_centers(queries, gallery, model, lambda_ref, rerank=False, gpu_index=0, verbose=1, sobel=False, grayscale=False):

    model.eval()
    queries_fvs = extractFeatures(queries, model, 500, gpu_index, sobel=sobel, grayscale=grayscale)
    gallery_fvs = extractFeatures(gallery, model, 500, gpu_index, sobel=sobel, grayscale=grayscale)

    queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
    gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

    centers = torch.load("centers_Duke.pt").cpu()
    print(centers.shape)
    
    tau = 0.04
    sim_queries = torch.mm(queries_fvs, centers.T)
    sim_queries = torch.exp(sim_queries)/torch.sum(torch.exp(sim_queries), dim=1, keepdim=True)

    #print(sim_queries[0])
    
    sim_gallery = torch.mm(gallery_fvs, centers.T)
    sim_gallery = torch.exp(sim_gallery)/torch.sum(torch.exp(sim_gallery), dim=1, keepdim=True)

    L1_dist = torch.cdist(sim_queries, sim_gallery, p=1.0)
    min_v = 1 - L1_dist/2
    max_v = 1 + L1_dist/2
    distmat = 1 - min_v/max_v
    print(torch.min(distmat, dim=1))
    distmat_ref = distmat.numpy()

    #exit()

    distmat_sim = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
    distmat_sim = distmat_sim.numpy()

    #distmat = distmat_sim + lambda_ref*distmat_ref
    distmat = distmat_ref

    #print(distmat)

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
        distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
        distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)

    del queries_fvs, gallery_fvs
    # Compute Ranks
    ranks=[1, 5, 10, 20]
    if verbose >= 1:
        print('Computing CMC and mAP ...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)

    #del distmat
    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
        
    return cmc[:20], mAP, distmat

def validateMultiPart(queries, gallery, model, rerank=False, gpu_index=0, verbose=1, sobel=False, grayscale=False):

    model.eval()
    queries_fvs_upper, queries_fvs_middle, queries_fvs_lower, queries_fvs = extractFeaturesMultiPart(queries, model, 500, 
                                                                                    gpu_index, sobel=sobel, grayscale=grayscale)
    gallery_fvs_upper, gallery_fvs_middle, gallery_fvs_lower, gallery_fvs = extractFeaturesMultiPart(gallery, model, 500, 
                                                                                    gpu_index, sobel=sobel, grayscale=grayscale)

    queries_fvs_upper = queries_fvs_upper/torch.norm(queries_fvs_upper, dim=1, keepdim=True)
    queries_fvs_middle = queries_fvs_middle/torch.norm(queries_fvs_middle, dim=1, keepdim=True)
    queries_fvs_lower = queries_fvs_lower/torch.norm(queries_fvs_lower, dim=1, keepdim=True)
    queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)

    gallery_fvs_upper = gallery_fvs_upper/torch.norm(gallery_fvs_upper, dim=1, keepdim=True)
    gallery_fvs_middle = gallery_fvs_middle/torch.norm(gallery_fvs_middle, dim=1, keepdim=True)
    gallery_fvs_lower = gallery_fvs_lower/torch.norm(gallery_fvs_lower, dim=1, keepdim=True)
    gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

    distmat_upper = torchreid.metrics.compute_distance_matrix(queries_fvs_upper, gallery_fvs_upper, metric="euclidean")
    distmat_upper = distmat_upper.numpy()

    distmat_middle = torchreid.metrics.compute_distance_matrix(queries_fvs_middle, gallery_fvs_middle, metric="euclidean")
    distmat_middle = distmat_middle.numpy()

    distmat_lower = torchreid.metrics.compute_distance_matrix(queries_fvs_lower, gallery_fvs_lower, metric="euclidean")
    distmat_lower = distmat_lower.numpy()

    distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
    distmat = distmat.numpy()

    #if rerank:
    #    print('Applying person re-ranking ...')
    #    distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
    #    distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
    #    distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)

    del queries_fvs_upper, queries_fvs_middle, queries_fvs_lower, queries_fvs
    del gallery_fvs_upper, gallery_fvs_middle, gallery_fvs_lower, gallery_fvs

    # Compute Ranks
    ranks=[1, 5, 10, 20]
    if verbose >= 1:
        print('Computing CMC and mAP Upper Body...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat_upper, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)

    #del distmat
    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))


    # Compute Ranks
    ranks=[1, 5, 10, 20]
    if verbose >= 1:
        print('Computing CMC and mAP Middle Body...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat_middle, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)

    #del distmat
    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

    # Compute Ranks
    ranks=[1, 5, 10, 20]
    if verbose >= 1:
        print('Computing CMC and mAP Lower Body...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat_lower, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)

    #del distmat
    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

    # Compute Ranks
    ranks=[1, 5, 10, 20]
    if verbose >= 1:
        print('Computing CMC and mAP Whole Body...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)

    #del distmat
    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

    distmat_all_parts = (distmat_upper + distmat_middle + distmat_lower + distmat)/4

     # Compute Ranks
    ranks=[1, 5, 10, 20]
    if verbose >= 1:
        print('Computing CMC and mAP All Body Parts ...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat_all_parts, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)

    #del distmat
    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
        
    return cmc[:20], mAP, distmat_all_parts

def validateDual(queries, gallery, model, rerank=False, gpu_index=0):
    model.eval()
    queries_fvs, queries_fvs_id, queries_fvs_bias = extractFeaturesDual(queries, model, 500, gpu_index)
    gallery_fvs, gallery_fvs_id, gallery_fvs_bias = extractFeaturesDual(gallery, model, 500, gpu_index)

    queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
    queries_fvs_id = queries_fvs_id/torch.norm(queries_fvs_id, dim=1, keepdim=True)
    queries_fvs_bias = queries_fvs_bias/torch.norm(queries_fvs_bias, dim=1, keepdim=True)

    gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)
    gallery_fvs_id = gallery_fvs_id/torch.norm(gallery_fvs_id, dim=1, keepdim=True)
    gallery_fvs_bias = gallery_fvs_bias/torch.norm(gallery_fvs_bias, dim=1, keepdim=True)

    distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
    distmat = distmat.numpy()

    distmat_id = torchreid.metrics.compute_distance_matrix(queries_fvs_id, gallery_fvs_id, metric="euclidean")
    distmat_id = distmat_id.numpy()

    distmat_bias = torchreid.metrics.compute_distance_matrix(queries_fvs_bias, gallery_fvs_bias, metric="euclidean")
    distmat_bias = distmat_bias.numpy()

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
        distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
        distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)


    # Compute Ranks
    ranks=[1, 5, 10, 20]
    print('Computing CMC and mAP ...')
    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)
    print('** Results **')
    print('mAP: {:.2%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))


    # Compute Ranks
    ranks=[1, 5, 10, 20]
    print('Computing CMC and mAP ...')
    cmc, mAP = torchreid.metrics.evaluate_rank(distmat_bias, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)
    print('** Results **')
    print('mAP: {:.2%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

    # Compute Ranks
    ranks=[1, 5, 10, 20]
    print('Computing CMC and mAP ...')
    cmc, mAP = torchreid.metrics.evaluate_rank(distmat_id, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)
    print('** Results **')
    print('mAP: {:.2%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
        
    return cmc[:20], mAP, distmat


def validateMultiView(queries, gallery, model, rerank=False, gpu_index=0):
	model.eval()
	queries_global_fvs, queries_spatial_fvs, queries_channel_fvs = extractFeaturesMultiView(queries, 
																							model, 500, 
																							gpu_index=gpu_index, 
																							eval_mode=True)

	gallery_global_fvs, gallery_spatial_fvs, gallery_channel_fvs = extractFeaturesMultiView(gallery, 
																							model, 500, 
																							gpu_index=gpu_index, 
																							eval_mode=True)

	queries_global_fvs = queries_global_fvs/torch.norm(queries_global_fvs, dim=1, keepdim=True)
	queries_spatial_fvs = queries_spatial_fvs/torch.norm(queries_spatial_fvs, dim=1, keepdim=True)
	queries_channel_fvs = queries_channel_fvs/torch.norm(queries_channel_fvs, dim=1, keepdim=True)

	gallery_global_fvs = gallery_global_fvs/torch.norm(gallery_global_fvs, dim=1, keepdim=True)
	gallery_spatial_fvs = gallery_spatial_fvs/torch.norm(gallery_spatial_fvs, dim=1, keepdim=True)
	gallery_channel_fvs = gallery_channel_fvs/torch.norm(gallery_channel_fvs, dim=1, keepdim=True)

	distmat_global = torchreid.metrics.compute_distance_matrix(queries_global_fvs, gallery_global_fvs, metric="euclidean").numpy()
	distmat_spatial = torchreid.metrics.compute_distance_matrix(queries_spatial_fvs, gallery_spatial_fvs, metric="euclidean").numpy()
	distmat_channel = torchreid.metrics.compute_distance_matrix(queries_channel_fvs, gallery_channel_fvs, metric="euclidean").numpy()
	
	distmat_ensemble = (distmat_global + distmat_spatial + distmat_channel)/3

	# Compute Ranks
	ranks=[1, 5, 10, 20]
	print('Computing CMC and mAP with ensembled features ...')
	cmc, mAP = torchreid.metrics.evaluate_rank(distmat_ensemble, queries[:,1], gallery[:,1], 
														queries[:,2], gallery[:,2], use_metric_cuhk03=False)
	print('** Results **')
	print('mAP: {:.2%}'.format(mAP))
	print('CMC curve')
	for r in ranks:
		print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

	return cmc[:20], mAP, distmat_ensemble


def calculateMetrics(distmat, queries, gallery):

	#compute Ranks
	ranks = [1,5,10]
	print('Computing CMC and mAP ...')
	cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
														queries[:,2], gallery[:,2], use_metric_cuhk03=False)

	print('** Results **')
	print('mAP: {:.2%}'.format(mAP))
	print('Ranks:')
	for r in ranks:
		print('Rank-{:<3}: {:.2%}'.format(r, cmc[r-1]))
	return cmc, mAP

def validatePRCC(queries, gallery, model, rerank=False, gpu_index=0):

	model.eval()

	num_queries = len(queries)
	num_galleries = len(gallery)

	queries_fvs = []
	for query_idx in range(num_queries):

		if query_idx == 0:
			print('Computing CMC and mAP for same-clothes evaluation...')
		elif query_idx == 1:
			print('Computing CMC and mAP for cross-clothes evaluation...')
		elif query_idx == 2:
			print('Computing CMC and mAP for both setups ...')

		all_mAPs = []
		all_cmcs = []
		for gal_idx in range(num_galleries):
			cmc, mAP, distmat = validate(queries[query_idx], gallery[gal_idx], model, gpu_index=gpu_index, verbose=0)
			all_mAPs.append(mAP)
			all_cmcs.append(cmc)

		mean_mAP = np.mean(all_mAPs)
		std_mAP = np.std(all_mAPs)

		mean_cmcs = np.mean(all_cmcs, axis=0)
		std_cmcs = np.std(all_cmcs, axis=0)

		ranks = [1,5,10]
		print('** Results **')
		print('mAP: {:.2%} +/- {:.2%}'.format(mean_mAP, std_mAP))
		print('Ranks:')
		for r in ranks:
			print('Rank-{:<3}: {:.2%} +/- {:.2%}'.format(r, mean_cmcs[r-1], std_cmcs[r-1]))
		

	return mean_cmcs[:20], mean_mAP, distmat


def validateDeepChange(queries, gallery, model, rerank=False, gpu_index=0, sobel=False, verbose=1, grayscale=False):

    model.eval()
    queries_fvs = extractFeatures(queries[0], model, 500, gpu_index, sobel=sobel, grayscale=grayscal)
    gallery_fvs = extractFeatures(gallery[0], model, 500, gpu_index, sobel=sobel, grayscale=grayscal)

    queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
    gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

    distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
    distmat = distmat.numpy()

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
        distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
        distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)

    del queries_fvs, gallery_fvs

    # Compute Ranks
    ranks=[1, 5, 10, 20]

    # Evaluation in cross-camera scenario
    if verbose >= 1:
        print('Computing CMC and mAP considering cross-camera ...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[0][:,1], gallery[0][:,1], 
                                                queries[0][:,2], gallery[0][:,2], use_metric_cuhk03=False)

    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

    # Evaluation in cross-days scenario
    if verbose >= 1:
        print('Computing CMC and mAP considering cross-days ...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[1][:,1], gallery[1][:,1], 
                                                queries[1][:,2], gallery[1][:,2], use_metric_cuhk03=False)

    #del distmat
    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))


    # Evaluation in cross-camera-and-days scenario
    if verbose >= 1:
        print('Computing CMC and mAP considering tracklets (original paper) ...')

    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[2][:,1], gallery[2][:,1], 
                                                queries[2][:,2], gallery[2][:,2], use_metric_cuhk03=False)

    #del distmat
    if verbose >= 1:
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

    return cmc[:20], mAP, distmat


class validation(object):


    def __init__(self, target, queries_images, gallery_images, evaluation_images=None, rerank=False, gpu_index=0, verbose=1, sobel=False, grayscale=False):

        self.target = target
        self.queries_images = queries_images
        self.gallery_images = gallery_images
        self.evaluation_images = evaluation_images
        self.rerank = rerank
        self.gpu_index = gpu_index
        self.verbose = verbose
        self.sobel = sobel
        self.grayscale = grayscale

        if target == "Places" or "VehicleID" in target:
            print("There are %d samples in evaluation" % len(evaluation_images))
            samples_labels = self.evaluation_images[:,1]
            unique_labels = np.unique(samples_labels)

            self.evaluation_folds = {0: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
                                1: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
                                2: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
                                3: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
                                4: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
                                5: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
                                6: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
                                7: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
                                8: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
                                9: [np.array([], dtype=np.int32), np.array([], dtype=np.int32)]}


            if target == "Places":

                kf = KFold(n_splits=10, shuffle=True, random_state=12)

                for label in unique_labels:
                    labels_idx = np.where(samples_labels == label)[0]
                    
                    for i, (gallery_index, query_index) in enumerate(kf.split(labels_idx)):

                        self.evaluation_folds[i][0] = np.append(self.evaluation_folds[i][0], labels_idx[query_index])
                        self.evaluation_folds[i][1] = np.append(self.evaluation_folds[i][1], labels_idx[gallery_index])

            elif "VehicleID" in target:

                for label in unique_labels:
                    labels_idx = np.where(samples_labels == label)[0]

                    if len(labels_idx) < 10:
                        selected_gallery_indexes = np.random.choice(labels_idx, size=10, replace=True)
                    else:
                        selected_gallery_indexes = np.random.choice(labels_idx, size=10, replace=False)
    
                    for i, gallery_index in enumerate(selected_gallery_indexes):

                        query_index = np.setdiff1d(labels_idx, gallery_index)

                        self.evaluation_folds[i][0] = np.append(self.evaluation_folds[i][0], query_index)
                        self.evaluation_folds[i][1] = np.append(self.evaluation_folds[i][1], gallery_index)

            else:
                raise NotImplementedError("Target dataset not implemented!")


    def validateCheckpoint(self, model):

        model.eval()
        evaluation_fvs = extractFeatures(self.evaluation_images, model, 500, self.gpu_index, sobel=self.sobel, grayscale=self.grayscale)
        evaluation_fvs = evaluation_fvs/torch.norm(evaluation_fvs, dim=1, keepdim=True)
        
        all_mAPs = []
        all_cmcs = []
        all_distmats = []

        if self.target == "Places":
            ranks = np.array([1, 5, 10, 20])
        else:
            ranks = np.array([1])

        for i in tqdm(np.arange(10)):
            probes_idx = self.evaluation_folds[i][0]
            gallery_idx = self.evaluation_folds[i][1]

            #print(probes_idx)

            #num_probes_labels = len(np.unique(evaluation[probes_idx][:,1]))
            #num_gallery_labels = len(np.unique(evaluation[gallery_idx][:,1]))

            #print("Fold %d" % i)
            #print("There are %d labels in probe set" % num_probes_labels)
            #print("There are %d labels in gallery set" % num_gallery_labels)
            #print("There are %d samples in common in probe and gallery sets" % len(np.intersect1d(probes_idx, gallery_idx)))


            queries_fvs = evaluation_fvs[probes_idx]
            gallery_fvs = evaluation_fvs[gallery_idx]       

            distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
            distmat = distmat.numpy()

            all_distmats.append(distmat)

        all_distmats = np.array(all_distmats)
        self.calculateMetrics(all_distmats)  
        return all_distmats

    def calculateMetrics(self, all_distmats):

        all_mAPs = []
        all_cmcs = []

        if self.target == "Places":
            ranks = np.array([1, 5, 10, 20])
        else:
            ranks = np.array([1, 5])

        for i in tqdm(np.arange(10)):

            probes_idx = self.evaluation_folds[i][0]
            gallery_idx = self.evaluation_folds[i][1]

            queries = self.evaluation_images[probes_idx]
            gallery = self.evaluation_images[gallery_idx]

            distmat = all_distmats[i]
        
            if self.rerank:
                print('Applying person re-ranking ...')
                distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
                distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
                distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)

            # Compute Ranks
            if self.verbose >= 1:
                print('Computing CMC and mAP ...')

            cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                        queries[:,2], gallery[:,2], use_metric_cuhk03=False)

            #del distmat
            if self.verbose >= 1:
                print('** Results **')
                print('mAP: {:.2%}'.format(mAP))
                print('CMC curve')
                for r in ranks:
                    print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

            all_mAPs.append(mAP)
            all_cmcs.append(cmc)

        mean_mAP = np.mean(all_mAPs)
        std_mAP = np.std(all_mAPs)

        mean_cmcs = np.mean(all_cmcs, axis=0)
        std_cmcs = np.std(all_cmcs, axis=0)

        print('** Results **')
        print('mAP: {:.2%} +/- {:.2%}'.format(mean_mAP, std_mAP))
        print('Ranks:')
        for r in ranks:
            print('Rank-{:<3}: {:.2%} +/- {:.2%}'.format(r, mean_cmcs[r-1], std_cmcs[r-1]))

        return mean_mAP, mean_cmcs
            


def validateOnDatasets(targets_names, queries_images_target, gallery_images_target, model_name, model_online, gpu_indexes, sobel=False):

	for target_idx in np.arange(len(targets_names)):
		target = targets_names[target_idx]
		print("Validating %s on %s ..." % (model_name, target))

		if target == "PRCC":
			cmc, mAP, distmat = validatePRCC(queries_images_target[target_idx], gallery_images_target[target_idx], model_online, 
		
        																									gpu_index=gpu_indexes[0])
		elif target == "VC-Clothes":
			print(colored("Validating with synthetic dataset ..."))
            #NSA
			cmc, mAP, distmat = validate(queries_images_target[target_idx][0], gallery_images_target[target_idx][0], model_online, 
																											gpu_index=gpu_indexes[0])
			print(colored("Validating with real dataset ..."))
			cmc, mAP, distmat = validate(queries_images_target[target_idx][1], gallery_images_target[target_idx][1], model_online, 
																											gpu_index=gpu_indexes[0])

		elif target == "DeepChange":
			cmc, mAP, distmat = validateDeepChange(queries_images_target[target_idx], gallery_images_target[target_idx], model_online, 
																											gpu_index=gpu_indexes[0], sobel=sobel)
		elif target == "ImageNet":

			# For ImageNet there are 10 gallery/query pairs
			number_of_galleries = len(gallery_images_target[target_idx])
			all_cmcs = []
			all_mAPs = []



            #NSA
			for pair_idx in range(number_of_galleries):
				print(colored("Validating on split %d ..." % (pair_idx+1), "white"))
				cmc, mAP, distmat = validate(queries_images_target[target_idx][pair_idx], 
												gallery_images_target[target_idx][pair_idx], 
												model_online, gpu_index=gpu_indexes[0])

				all_mAPs.append(mAP)
				all_cmcs.append(cmc)

			mean_mAP = np.mean(all_mAPs)
			std_mAP = np.std(all_mAPs)

			mean_cmcs = np.mean(all_cmcs, axis=0)
			std_cmcs = np.std(all_cmcs, axis=0)

			ranks = [1,5,10]
			print('** Results **')
			print('mAP: {:.2%} +/- {:.2%}'.format(mean_mAP, std_mAP))
			print('Ranks:')
			for r in ranks:
				print('Rank-{:<3}: {:.2%} +/- {:.2%}'.format(r, mean_cmcs[r-1], std_cmcs[r-1]))
	
		else:
			cmc, mAP, distmat = validate(queries_images_target[target_idx], gallery_images_target[target_idx], 
																	model_online, gpu_index=gpu_indexes[0], sobel=sobel)

	return cmc, mAP, distmat