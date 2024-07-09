import torch
import numpy as np

import faiss
import time

from getFeatures import extractFeatures

from termcolor import colored
from sklearn.cluster import cluster_optics_xi, OPTICS, DBSCAN
from sklearn.metrics import silhouette_samples
import joblib

from scipy import stats

import OptimizedReRanking
#from PEACH import PEACH

# fvs must have norm 1 (projected on the radius-one hypersphere)
def GPUOPTICS(fvs, images, k, minPts=4, xi=0.05):

	#fvs = torch.rand(300,2)
	#fvs = fvs/torch.norm(fvs, dim=1, keepdim=True)

	xb = fvs.numpy()
	assert xb.dtype == 'float32' # faiss only holds float32 arrays

	nb, d = xb.shape
	cpu_index = faiss.IndexFlatL2(d)
	gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

	gpu_index.add(xb)
	print("There are %d feature vectors!" % gpu_index.ntotal)

	t0 = time.time()
	D, I = gpu_index.search(xb, k)
	D = D**0.5
	D, I = ReRankingV2(D, I)
	#D, I = ReRanking(D, I)
	#D, I = Refinement(D, I)
	tf = time.time()
	print("Min Dist:", D[:,1:].min())
	print("kNN search done in %f seconds!" % (tf-t0))
	
	t0 = time.time()
	core_distances = D[:, (minPts-1)]

	samples_distances = 4*np.ones(nb)
	samples_idx_on_heap = -1*np.ones(nb, dtype=int)

	heap = []
	heap_size = 0

	predecessors = np.zeros(nb, dtype=int)
	visited = np.zeros(nb)

	#vertex_id = 0
	#samples_idx_on_heap[vertex_id] = -2 

	reachability_plot = []
	ordering_vertices = []
	
	#there_are_unvisited_vertices = True

	for full_index_view in range(nb):

		if visited[full_index_view] == 0:

			reachability_plot.append(np.inf)
			ordering_vertices.append(full_index_view)

			vertex_id = full_index_view
			there_are_unvisited_vertices = True

			predecessors[vertex_id] = -1
			samples_idx_on_heap[vertex_id] = -2

			while there_are_unvisited_vertices:

				visited[vertex_id] = 1
				for j in range(1,k):
					neighbor_index = I[vertex_id,j]
					reach_dist = max(core_distances[vertex_id], D[vertex_id,j])
					if reach_dist < samples_distances[neighbor_index] and visited[neighbor_index] == 0:

						samples_distances[neighbor_index] = reach_dist

						if samples_idx_on_heap[neighbor_index] == -1:
							element = [neighbor_index, reach_dist]
							heap, heap_size, samples_idx_on_heap = addElementOnHeap(heap, heap_size, element, samples_idx_on_heap)
						else:
							element_idx = samples_idx_on_heap[neighbor_index]
							heap[element_idx][1] = reach_dist
							heap, samples_idx_on_heap = updateElementOnHeap(heap, heap_size, element_idx, samples_idx_on_heap)
							

						predecessors[neighbor_index] = vertex_id

				if len(heap) > 0:
					minElement, heap, heap_size, samples_idx_on_heap = getMinAndUpdateHeap(heap, heap_size, samples_idx_on_heap)
					vertex_id, vertex_distance = minElement
					reachability_plot.append(vertex_distance)
					ordering_vertices.append(vertex_id)
				else:
					there_are_unvisited_vertices = False


	assert np.sum(visited) == nb
	reachability_plot = np.array(reachability_plot)
	ordering_vertices = np.array(ordering_vertices)
	#predecessors[0] = -1

	print(len(reachability_plot), len(ordering_vertices), len(predecessors))

	tf = time.time()
	print("Reachability plot construction done in %f seconds!" % (tf-t0))

	t0 = time.time()
	original_reachability_idx = np.argsort(ordering_vertices)
	original_reachability = reachability_plot[original_reachability_idx]

	#xi_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,1e-3, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
	#for xi in xi_values:
	pseudo_labels, _ = cluster_optics_xi(reachability=original_reachability, predecessor=predecessors, ordering=ordering_vertices, 
																		min_samples=minPts, xi=xi, predecessor_correction=True)

	#	clusters = np.unique(pseudo_labels)
	#	if -1 in clusters:
	#		num_clusters = len(clusters) - 1
	#	else:
	#		num_clusters = len(clusters)
	#	print("Xi value: %.10f, number of clusters: %d" % (xi, num_clusters))

	#	selected_images = images[pseudo_labels != -1]
	#	proposed_labels = pseudo_labels[pseudo_labels != -1]

	#	ratio_of_reliable_data = proposed_labels.shape[0]/pseudo_labels.shape[0]
	#	print("Reliability after clustering: %.3f" % ratio_of_reliable_data)
	
	tf = time.time()
	print("Pseudo labels obtained in %f seconds!" % (tf-t0))

	# Sklearn OPTICS for validation
	#clustering = OPTICS(min_samples=minPts, cluster_method='xi', xi=xi, metric='euclidean', 
	#																	predecessor_correction=True, n_jobs=-1).fit(xb)

	#print(1,original_reachability, 2,clustering.reachability_, "diff:", np.sum(original_reachability[1:] - clustering.reachability_[1:]))
	#print(1,ordering_vertices, 2,clustering.ordering_, "diff:", np.sum(ordering_vertices - clustering.ordering_))
	#print(1, predecessors, 2, clustering.predecessor_, "diff:", np.sum(predecessors - clustering.predecessor_))
	#print(1, pseudo_labels, 2, clustering.labels_, "diff:", np.sum(pseudo_labels - clustering.labels_))
	
	#selected_images = images[pseudo_labels != -1]
	#proposed_labels = pseudo_labels[pseudo_labels != -1]

	#ratio_of_reliable_data = proposed_labels.shape[0]/pseudo_labels.shape[0]
	#print("Reliability after clustering: %.3f" % ratio_of_reliable_data)
	#exit()
	return pseudo_labels


# fvs must have norm 1 (projected on the radius-one hypersphere)
def GPUDBSCAN(D, I, k, minPts=4, eps=0.5, verbose=1):

	t0 = time.time()
	nb = D.shape[0]
	pseudo_labels = -np.ones(nb, dtype=int)
	seen_points = np.zeros(nb, dtype=int)

	distance_mask = D <= eps
	number_of_closest_neighbors_within_eps_distance = np.sum(distance_mask, axis=1)
	mask_core_points = number_of_closest_neighbors_within_eps_distance >= minPts
	core_points_idx = np.where(mask_core_points)[0]

	#print(core_points_idx.shape)
	current_label = 0
	core_points_stack = []

	for outer_cp_idx in core_points_idx:

		assert len(core_points_stack) == 0

		if pseudo_labels[outer_cp_idx] == -1:
			pseudo_labels[outer_cp_idx] = current_label
			current_label += 1

			core_points_stack.append(outer_cp_idx)
			seen_points[outer_cp_idx] = 1
			
			while len(core_points_stack) > 0:

				cp_idx = core_points_stack.pop(0)
				core_point_label = pseudo_labels[cp_idx]
				#print("Core point label: %d" % core_point_label)
				#print(cp_idx, I[cp_idx], D[cp_idx])
				for j in np.arange(k):
					neighbor_idx = I[cp_idx][j]
					#print("Neighbor idx", neighbor_idx)
					if distance_mask[cp_idx][j] and seen_points[neighbor_idx] == 0:
						#print("Not seen!")
						pseudo_labels[neighbor_idx] = core_point_label
						seen_points[neighbor_idx] = 1
						if mask_core_points[neighbor_idx]:
							#print("It is a core point!")
							core_points_stack.append(neighbor_idx)


	present_pseudo_labels, freqs = np.unique(pseudo_labels, return_counts=True)

	if present_pseudo_labels[0] == -1:
		start = 1
	else:
		start = 0

	undersize_clusters_idx = np.where(freqs[start:] < minPts)[0]
	undersize_clusters = present_pseudo_labels[start:][undersize_clusters_idx]

	for label in undersize_clusters:
		pseudo_labels[pseudo_labels == label] = -1


	#print(np.unique(pseudo_labels, return_counts=True)[1][1:].min(),current_label)
	#selected_images = images[non_outliers_indexes]
	#proposed_labels = pseudo_labels[non_outliers_indexes]

	#ratio_of_reliable_data = proposed_labels.shape[0]/pseudo_labels.shape[0]
	#print("Reliability after clustering: %.3f" % ratio_of_reliable_data)
	#exit()
	tf = time.time()

	if verbose >= 1:
		print("Clustering done in %f seconds!" % (tf-t0))

	return pseudo_labels, mask_core_points #, ratio_of_reliable_data


def defineEpsForDBSCAN(fvs, k, minPts, iteration_number, verbose=1, 
														single_gpu_for_clustering=False, v="model_name"):

	#fvs = torch.rand(1000,2)
	#fvs = fvs/torch.norm(fvs, dim=1, keepdim=True)

	xb = fvs.numpy()
	assert xb.dtype == 'float32' # faiss only holds float32 arrays
	nb, d = xb.shape
	cpu_index = faiss.IndexFlatL2(d)

	if single_gpu_for_clustering:
		print("Single GPU just for clustering")
		res = faiss.StandardGpuResources()
		gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
	else:
		print("Multiple GPUs for clustering")
		gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

	gpu_index.add(xb)
	if verbose >= 1:
		print("There are %d feature vectors!" % gpu_index.ntotal)

	'''
	t0 = time.time()
	D, I = gpu_index.search(xb, k)
	D = D**0.5
	D, I = ReRankingV2(D, I)
	#D, I = Refinement(D, I)
	tf = time.time()
	print(I)
	print("Python ReRanking time: %f seconds" % (tf-t0))
	'''


	t0 = time.time()
	D, I = gpu_index.search(xb, k)
	D = D**0.5
	D, I = OptimizedReRanking.ReRankingV2(D, np.int32(I))
	#D, I = Refinement(D, I)
	tf = time.time()
	
	if verbose >= 1:
		print("Min Dist:", D[:,1:].min())
		print("Max Dist:", D[:,1:].max())
		print("kNN search done in %f seconds!" % (tf-t0))
	
	fvs_memory = xb.nbytes/1e6
	distance_matrix_memory = D.nbytes/1e6
	total_memory = fvs_memory + distance_matrix_memory

	#if verbose >= 1:
	#	print(colored("Memory footprint: %.7f (total size) = %.7f (fvs size) + %.7f (distance matrix size)" % (total_memory, fvs_memory, distance_matrix_memory), "green"))

	#t0 = time.time()

	#eps_mean = np.mean(D[:,(minPts-1)])
	#eps_median = np.median(D[:,(minPts-1)])
	
	#if eps_mean >= eps_median:
	#	min_eps = eps_median
	#	max_eps = eps_mean
	#else:
	#	min_eps = eps_mean
	#	max_eps = eps_median

	#eps_values = np.linspace(min_eps,max_eps,num=5)
	#mi = D[:,(minPts-1)].mean()
	#sigma = D[:,(minPts-1)].std()
	#eps = min(mi/(mi + sigma), mi + sigma)
	#eps = mi/(1.0 + sigma)
	#print("Calculated Mean/Sigma eps: %.3f" % eps)
	#print(colored("Eps value on DBSCAN clustering: %f" % eps, "cyan"))
	#eps = np.mean(D[:,99])/np.max(D[:,99])
	#print("Calculated eps EVT: %.3f" % eps)

	# This fits a weilbull distribution
	#alpha, shape_k, loc, scale = stats.exponweib.fit(D[:,(minPts-1)], f0=1, floc=0)
	#t = range_percentile[iteration_number-1]
	#eps = scale*((-np.log(1-t))**(1/shape_k)) # inverse CDF of Weilbull distribution
	#print("shape: %.2f, scale: %.2f, quantile: %.2f, Eps: %.2f" % (shape_k, scale, t, eps))


	#q = range_percentile[iteration_number-1]
	#eps = np.quantile(D[:,(minPts-1)], q)
	#print("Quantile: %.2f, Eps value: %.2f" % (q,eps)) 



	#if verbose >= 1:
	#	print("alpha: %.2f, shape: %.2f, loc: %.2f, scale: %.2f" % (alpha, shape_k, loc, scale))
	#	print("Mean Weibull", stats.exponweib.mean(1, shape_k, loc=0, scale=1))
	#	print("Median Weibull", stats.exponweib.median(1, shape_k, loc=0, scale=1))

	#eps = stats.exponweib.median(1, shape_k, loc=0, scale=1)

	#if verbose >= 1:
	#	print(colored("Eps value on DBSCAN clustering: %f" % eps, "cyan"))

	#joblib.dump(D[:,(minPts-1)], "minPts_distances/dist1_neighbors_%s_%d.pkl" % (v, iteration_number))
	#exit()
	
	# just to return some eps value. It is not used
	eps = 0.5 
	return eps, D, I


def selfAdaptiveGPUDBSCAN(model, images, fvs, k, tau_q, eps=None, minPts=4, gpu_index=0):

	data_driven_eps, D, I = defineEpsForDBSCAN(fvs, k, minPts)
	print("Min Pts:", minPts)

	if eps:
		print("Eps:", eps)
		pseudo_labels = GPUDBSCAN(D, I, images, k, minPts=minPts, eps=eps)
	else:
		print("Eps:", data_driven_eps)
		pseudo_labels = GPUDBSCAN(D, I, images, k, minPts=minPts, eps=data_driven_eps)

	unique_pseudo_labels, clusters_sizes = np.unique(pseudo_labels, return_counts=True)
	if unique_pseudo_labels[0] == -1:
		j = 1
	else:
		j = 0

	print("Frequencies on general Clustering:", clusters_sizes.shape[0])

	last_label = unique_pseudo_labels[j:][-1] + 1
	smallest_cluster_size = np.min(clusters_sizes[j:])
	biggest_cluster_size = np.max(clusters_sizes[j:])
	current_eps = eps

	print("Smallest cluster size:", smallest_cluster_size, "Biggest cluster size:", biggest_cluster_size)
	print("Last label:", last_label, "current eps:", current_eps)

	'''
	iter_idx = 1
	while True:
		#print(colored("#============ New Iteration %d ============#" % iter_idx, "yellow"))
		non_outliers_indexes = np.where(pseudo_labels != -1)[0]
		selected_labels = pseudo_labels[non_outliers_indexes]
		selected_images = images[non_outliers_indexes]
		selected_fvs = fvs[non_outliers_indexes]

		low_quality_clusters = SelectClustersByQuality(selected_images, selected_labels, model, tau_q=tau_q, gpu_index=gpu_index, verbose=0)

		#print("Low Quality Clusters", low_quality_clusters)

		if len(low_quality_clusters) == 0:
			break

		current_eps = current_eps*0.99
		print("Current Eps", current_eps)

		for low_label in low_quality_clusters:

			cluster_size = np.sum(pseudo_labels == low_label)
			#print("Cluster size", cluster_size)
			
			if cluster_size < 10*smallest_cluster_size:
				pseudo_labels[pseudo_labels == low_label] = -1
			else:
				cluster_images = selected_images[selected_labels == low_label]
				#print(cluster_images)
				cluster_fvs = selected_fvs[selected_labels == low_label]
				#print(cluster_fvs.shape)
				_, D_cluster, I_cluster = defineEpsForDBSCAN(cluster_fvs, k, verbose=0)
				cluster_pseudo_labels = GPUDBSCAN(D_cluster, I_cluster, cluster_images, k, NN_idx=4, minPts=minPts, 
																									eps=current_eps, verbose=0)

				non_outliers_indexes = np.where(cluster_pseudo_labels != -1)[0]
				#print(cluster_pseudo_labels, non_outliers_indexes)
				cluster_pseudo_labels[non_outliers_indexes] += last_label
				#print(cluster_pseudo_labels)
				pseudo_labels[pseudo_labels == low_label] = cluster_pseudo_labels

				unique_cluster_pseudo_labels = np.unique(cluster_pseudo_labels)
				if unique_cluster_pseudo_labels[0] == -1:
					j = 1
				else:
					j = 0

				last_label = unique_cluster_pseudo_labels[j:][-1] + 1
				#print("New Last label", last_label)

		iter_idx += 1

	'''

	'''
	selected_images = images[pseudo_labels != -1]
	proposed_labels = pseudo_labels[pseudo_labels != -1]

	ratio_of_reliable_data = proposed_labels.shape[0]/pseudo_labels.shape[0]
	print("Reliability after clustering: %.3f" % ratio_of_reliable_data)
	
	return selected_images, proposed_labels, ratio_of_reliable_data
	'''

	return pseudo_labels

def EnsembleGPUDBSCAN(images, fvs, k, eps_values, minPts=4, gpu_index=0):

	_, D, I = defineEpsForDBSCAN(fvs, k, minPts)
	print("Min Pts:", minPts)
	print("Eps values:", eps_values)
	N, kNN = D.shape

	pseudo_labels_along_clustering = []
	core_points_along_clustering = []

	for eps in eps_values:
		pseudo_labels, core_points = GPUDBSCAN(D, I, images, k, minPts=minPts, eps=eps)
		pseudo_labels_along_clustering.append(pseudo_labels)
		core_points_along_clustering.append(core_points)

	pseudo_labels_along_clustering = np.array(pseudo_labels_along_clustering)	
	core_points_along_clustering = np.array(core_points_along_clustering)

	last_clustering_pseudo_labels = pseudo_labels_along_clustering[-1]
	last_clustering_core_points = core_points_along_clustering[-1]

	number_of_runs = len(eps_values)
	selecting_mask = np.ones(N)

	for i in range(N):
		if -1 in pseudo_labels_along_clustering[:,i]:

			#how_many_times_was_outlier = np.where(pseudo_labels_along_clustering[:,i] == -1)[0].shape[0]
			#percentage = how_many_times_was_outlier/number_of_runs

			#if percentage >= 0.5:
			selecting_mask[i] = 0

	# To apply mask, we let all labels between 0 <= label <= N (outliers are 0 after transformation)
	last_clustering_pseudo_labels = last_clustering_pseudo_labels + 1
	# Apply mask letting only samples that most of the time have not been outliers
	last_clustering_pseudo_labels = last_clustering_pseudo_labels*selecting_mask
	last_clustering_core_points = last_clustering_core_points*selecting_mask
	# Let labels again in -1 <= label <= N-1 range where -1 are again the outliers
	last_clustering_pseudo_labels = last_clustering_pseudo_labels - 1

	unique_pseudo_labels, clusters_sizes = np.unique(last_clustering_pseudo_labels, return_counts=True)
	if unique_pseudo_labels[0] == -1:
		j = 1
	else:
		j = 0

	print("Number of clusters:", (clusters_sizes.shape[0]-j))

	last_label = unique_pseudo_labels[j:][-1] + 1
	smallest_cluster_size = np.min(clusters_sizes[j:])
	biggest_cluster_size = np.max(clusters_sizes[j:])
	current_eps = eps

	print("Smallest cluster size:", smallest_cluster_size, "Biggest cluster size:", biggest_cluster_size)
	print("Last label:", last_label, "current eps:", current_eps)

	return last_clustering_pseudo_labels, last_clustering_core_points

def clusterFainessGPUDBSCAN(images, fvs, k, minPts=4, gpu_index=0):

	_, D, I = defineEpsForDBSCAN(fvs, k, minPts)
	print("Min Pts:", minPts)
	N, kNN = D.shape

	pseudo_labels_along_clustering = []
	core_points_along_clustering = []
	best_fairness = 0.0

	eps_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
	print("Eps values:", eps_values)
	for eps in eps_values:

		print(colored("#======== Eps: %.2f ========#" % eps, "yellow"))
		pseudo_labels, core_points = GPUDBSCAN(D, I, images, k, minPts=minPts, eps=eps)
		unique_pseudo_labels, clusters_sizes = np.unique(pseudo_labels, return_counts=True)
		if unique_pseudo_labels[0] == -1:
			j = 1
		else:
			j = 0

		if len(clusters_sizes[j:]) > 0:
			smallest_cluster_size = np.min(clusters_sizes[j:])
			biggest_cluster_size = np.max(clusters_sizes[j:])
			#std_cluster_size = np.std(clusters_sizes[j:])
			
			cluster_fainess_level = np.sum(clusters_sizes[j:]**0.5)/((biggest_cluster_size-smallest_cluster_size)+1e-7)

			print("Number of clusters:", clusters_sizes.shape[0]-1)
			print("Smallest cluster size:", smallest_cluster_size, "Biggest cluster size:", biggest_cluster_size)
			#print("Standard deviation cluster size:", std_cluster_size)
			print("Cluster Fainess level: %.7f" % cluster_fainess_level)

			#print(np.unique(pseudo_labels, return_counts=True)[1])

			if cluster_fainess_level > best_fairness:
				best_fairness = cluster_fainess_level
				best_eps = eps
				best_pseudo_labels = pseudo_labels
				best_core_points = core_points
		else:
			print("This epsilon value is not high enough to create clusters. Just non-core points have been found.")
		
	print(colored("Selected eps: %.2f" % best_eps, "green"))
	return best_pseudo_labels


def SelectClustersByQuality(images, pseudo_labels, model, tau_q=0.35, gpu_index=0, verbose=1):


	fvs = extractFeatures(images, model, 500, gpu_index=gpu_index)
	fvs = fvs/torch.norm(fvs, dim=1, keepdim=True)

	unique_pseudo_labels = np.unique(pseudo_labels)

	if unique_pseudo_labels[0] == -1:
		j = 1
	else:
		j = 0

	# Computer Silhouette
	#silhouette_scores = silhouette_samples(fvs, pseudo_labels, metric='euclidean')
	mean_vectors = []
	all_sparseness = []

	if verbose >= 1:
		print("Num classes: %d" % len(unique_pseudo_labels[j:]))

	for lab in unique_pseudo_labels[j:]:
		#print(colored("Statistics on Cluster %d" % lab, "green"))
		cluster_fvs = fvs[pseudo_labels == lab]
		#cluster_silhouette = silhouette_scores[pseudo_labels == lab].mean()
		mean_cluster_fvs = torch.mean(cluster_fvs, dim=0, keepdims=True)
		mean_cluster_fvs = mean_cluster_fvs/torch.norm(mean_cluster_fvs, dim=1, keepdim=True)
		cluster_sparseness = torch.cdist(mean_cluster_fvs, cluster_fvs, p=2.0).mean()
		all_sparseness.append(cluster_sparseness)

		if len(mean_vectors) == 0:
			mean_vectors = mean_cluster_fvs
		else:
			mean_vectors = torch.cat((mean_vectors, mean_cluster_fvs), dim=0)

	all_sparseness = torch.Tensor(all_sparseness)
	distance_means = torch.cdist(mean_vectors, mean_vectors, p=2.0)
	distance_to_closest_cluster = torch.topk(distance_means, k=2, dim=1, largest=False)[0][:,1]

	cluster_silhouette_modified = (distance_to_closest_cluster - all_sparseness)/torch.maximum(distance_to_closest_cluster, all_sparseness)
	#print(cluster_silhouette_modified, cluster_silhouette_modified.min(), cluster_silhouette_modified.max())
	quality_score = (1.0 - (all_sparseness/2.0))*((cluster_silhouette_modified+1)/2.0)

	mean_quality_score = quality_score.mean().item()
	median_quality_score = torch.median(quality_score).item()
	min_quality_score = quality_score.min().item()
	max_quality_score = quality_score.max().item()

	if verbose >= 1:
		print("Mean Quality Score: %.2f" % mean_quality_score)
		print("Median Quality Score: %.2f" % median_quality_score)
		print("Min Quality Score: %.2f" % min_quality_score)
		print("Max Quality Score: %.2f" % max_quality_score)

	low_quality_clusters = []
	lab_idx = 0
	for lab in unique_pseudo_labels[j:]:
		qscore = quality_score[lab_idx]
		if qscore <= tau_q:
			low_quality_clusters.append(lab)

		lab_idx += 1

	return low_quality_clusters

def ReRankingV2(D, I):

	print("Performing Re-Ranking...")
	RD = D.copy()

	n, k = D.shape
	print(n,k)

	expD = np.exp(-D)

	for sample_idx in np.arange(n):
		sample_neighbors = I[sample_idx]
		for j in np.arange(k):
			neighbor_index = I[sample_idx, j]
			neigh_neighbors = I[neighbor_index]

			if neighbor_index > sample_idx:

				if sample_idx in I[neighbor_index]:

					min_sum = 0
					max_sum = 0

					#print(colored("#======== Nossa Senhora Aparecida ========#", "yellow"))
					#print("Sample idx:", sample_idx)
					#print("Sample neighbor:", sample_neighbors)
					#print("Sample distance neighbors:", expD[sample_idx])

					#print("Neighbor idx:", neighbor_index)
					#print("Neigh neighboors:", neigh_neighbors)
					#print("Neigbors distance neigbors:", expD[neighbor_index])

					elements, idx_on_first, idx_on_second = np.intersect1d(sample_neighbors,neigh_neighbors,return_indices=True)
					idx_only_on_first = np.setdiff1d(np.arange(k), idx_on_first)
					idx_only_on_second = np.setdiff1d(np.arange(k), idx_on_second)

					#print("Elements in common:", elements)
					#print("First element idx:", idx_on_first)
					#print("Second element idx:", idx_on_second)
					#print("Idx only on first:", idx_only_on_first)
					#print("Idx only on second:", idx_only_on_second)

					min_sum += np.minimum(expD[sample_idx, idx_on_first], expD[neighbor_index, idx_on_second]).sum()
					max_sum += np.maximum(expD[sample_idx, idx_on_first], expD[neighbor_index, idx_on_second]).sum()

					#print("Min and Max sum:", min_sum, max_sum)

					max_sum += expD[sample_idx, idx_only_on_first].sum() + expD[neighbor_index, idx_only_on_second].sum()

					#print("Max sum:", max_sum)

					J = 1 - (min_sum/max_sum)

					if J > 1.0:
						print(min_sum, max_sum)
						print(expD[sample_idx])
						print(expD[neighbor_index])
						exit()
					#print(D[sample_idx, j], J)
					RD[sample_idx, j] = J
					#print("Jaccard distance:", J)
					idx = np.where(I[neighbor_index] == sample_idx)[0]
					#print("Localition of sample in mutual neighborhood", idx)
					if len(idx) > 0:
						RD[neighbor_index, idx] = J

				else:
					RD[sample_idx, j] = 1.0

			else:

				if not sample_idx in I[neighbor_index]:
					RD[sample_idx, j] = 1.0


		#exit()

	sorted_idx_by_row = np.argsort(RD, axis=1)
	RD = np.array([RD[i,sorted_idx_by_row[i]] for i in range(RD.shape[0])])
	RI = np.array([I[i,sorted_idx_by_row[i]] for i in range(I.shape[0])])
	#NSA
	return RD, RI


def ReRanking(D, I):

	print("Performing Re-Ranking...")
	RD = D.copy()

	n, k = D.shape
	print(n,k)

	for sample_idx in np.arange(n):
		sample_neighbors = I[sample_idx]
		for j in np.arange(k):
			neighbor_index = I[sample_idx, j]
			if sample_idx in I[neighbor_index] and neighbor_index > sample_idx:
				#print(D[sample_idx], D[neighbor_index])
				min_sum = 0
				max_sum = 0

				neigh_neighbors = I[neighbor_index]
				#print(sample_neighbors, neigh_neighbors)
				joined_neighbors = np.concatenate((sample_neighbors, neigh_neighbors))
				#print(joined_neighbors)
				sorted_values = np.sort(joined_neighbors, kind='mergesort')
				#print(sorted_values)
				sorted_idx = np.argsort(joined_neighbors, kind='mergesort')
				#print(sorted_idx)
				for nidx in np.arange(2*k):
					 
					if nidx < 2*k-1 and sorted_values[nidx] == sorted_values[nidx+1]:
						#print("%d is in both!" % sorted_values[nidx])

						# Be careful with sorting algorithm! Here we assume all indexes below k are neighbors of
						# the samples, and indexes greater or equal to k are neighbor of the sample neighbor!
						if sorted_idx[nidx] >= k:
							sample_neighbor_idx = sorted_idx[nidx] - k 
						else:
							sample_neighbor_idx = sorted_idx[nidx]

						if sorted_idx[nidx+1] >= k:
							neigh_neighbor_idx = sorted_idx[nidx+1] - k
						else:
							neigh_neighbor_idx = sorted_idx[nidx+1]

						#print(sample_neighbor_idx, neigh_neighbor_idx)
						#print(D[sample_idx, sample_neighbor_idx], D[neighbor_index, neigh_neighbor_idx])

						min_sum += min(np.exp(-D[sample_idx, sample_neighbor_idx]), np.exp(-D[neighbor_index, neigh_neighbor_idx]))
						max_sum += max(np.exp(-D[sample_idx, sample_neighbor_idx]), np.exp(-D[neighbor_index, neigh_neighbor_idx]))

					elif nidx == 0 or sorted_values[nidx-1] != sorted_values[nidx]:

						if sorted_idx[nidx] >= k:
							neigh_neighbor_idx = sorted_idx[nidx] - k 
							#print("neigh", neigh_neighbor_idx)
							#print(D[neighbor_index, neigh_neighbor_idx], 2.0)

							max_sum += np.exp(-D[neighbor_index, neigh_neighbor_idx])
							#min_sum += np.exp(-2.0)

						else:
							sample_neighbor_idx = sorted_idx[nidx]
							#print("sample", sample_neighbor_idx)
							#print(D[sample_idx, sample_neighbor_idx], 2.0)

							max_sum += np.exp(-D[sample_idx, sample_neighbor_idx])
							#min_sum += np.exp(-2.0)


				
				#print(min_sum, max_sum)
				J = 1 - (min_sum/max_sum)
				#print(D[sample_idx, j], J)
				RD[sample_idx, j] = J

				idx = np.where(I[neighbor_index] == sample_idx)[0]
				RD[neighbor_index, idx] = J

				#print(RD[sample_idx, j], RD[neighbor_index, idx])
				#exit()

			elif sample_idx not in I[neighbor_index]:
				RD[sample_idx, j] = 1.0


	sorted_idx_by_row = np.argsort(RD, axis=1)
	RD = np.array([RD[i,sorted_idx_by_row[i]] for i in range(RD.shape[0])])
	RI = np.array([I[i,sorted_idx_by_row[i]] for i in range(I.shape[0])])
	
	return RD, RI



def Refinement(D, I):

	RD = D.copy()

	n, k = D.shape
	print(n,k)

	for sample_idx in np.arange(n):
		sample_neighbors = I[sample_idx]
		for j in np.arange(k):
			neighbor_index = I[sample_idx, j]
			if sample_idx not in I[neighbor_index]:
				RD[sample_idx, j] = 2.0


	sorted_idx_by_row = np.argsort(RD, axis=1)
	RD = np.array([RD[i,sorted_idx_by_row[i]] for i in range(RD.shape[0])])
	RI = np.array([I[i,sorted_idx_by_row[i]] for i in range(I.shape[0])])
	
	return RD, RI

def addElementOnHeap(heap, heap_size, element, samples_idx_on_heap):


	heap.append(element)
	heap_size += 1
	element_idx = heap_size-1
	keep_updating_heap = True

	if heap_size > 1:
		while keep_updating_heap:
			if element_idx % 2 == 0:
				parent_idx = (element_idx - 2)//2
			else:
				parent_idx = (element_idx - 1)//2

			parent_element = heap[parent_idx]

			if parent_element[1] > heap[element_idx][1] or (parent_element[1] == heap[element_idx][1] and parent_element[0] > heap[element_idx][0]):
				heap[parent_idx] = heap[element_idx]
				heap[element_idx] = parent_element
				samples_idx_on_heap[parent_element[0]] = element_idx
				element_idx = parent_idx
			else:
				keep_updating_heap = False

			if element_idx == 0:
				keep_updating_heap = False

	samples_idx_on_heap[heap[element_idx][0]] = element_idx
	return heap, heap_size, samples_idx_on_heap

def updateElementOnHeap(heap, heap_size, element_idx, samples_idx_on_heap):

	
	if element_idx > 0:

		keep_updating_heap = True
		while keep_updating_heap:
			if element_idx % 2 == 0:
				parent_idx = (element_idx - 2)//2
			else:
				parent_idx = (element_idx - 1)//2

			parent_element = heap[parent_idx]

			if parent_element[1] > heap[element_idx][1] or (parent_element[1] == heap[element_idx][1] and parent_element[0] > heap[element_idx][0]):
				heap[parent_idx] = heap[element_idx]
				heap[element_idx] = parent_element
				samples_idx_on_heap[parent_element[0]] = element_idx
				element_idx = parent_idx
			else:
				keep_updating_heap = False

			if element_idx == 0:
				keep_updating_heap = False

	samples_idx_on_heap[heap[element_idx][0]] = element_idx
	return heap, samples_idx_on_heap

def getMinAndUpdateHeap(heap, heap_size, samples_idx_on_heap):


	minElement = heap[0]
	samples_idx_on_heap[minElement[0]] = -2

	heap[0] = heap[-1]
	del heap[-1]
	heap_size -= 1 

	element_idx = 0
	keep_updating_heap = True

	while keep_updating_heap:

		left_son_idx = 2*element_idx + 1
		right_son_idx = 2*element_idx + 2

		if left_son_idx < heap_size and right_son_idx < heap_size:
			left_son = heap[left_son_idx]
			right_son = heap[right_son_idx]

			if left_son[1] < right_son[1] or (left_son[1] == right_son[1] and left_son[0] < right_son[0]):
				selected_element = heap[left_son_idx]
				selected_son_idx = left_son_idx
			elif left_son[1] > right_son[1] or (left_son[1] == right_son[1] and left_son[0] > right_son[0]):
				selected_element = heap[right_son_idx]
				selected_son_idx = right_son_idx

		elif left_son_idx < heap_size:
			selected_element = heap[left_son_idx]
			selected_son_idx = left_son_idx

		elif right_son_idx < heap_size:
			selected_element = heap[right_son_idx]
			selected_son_idx = right_son_idx
		else:
			keep_updating_heap = False

		if keep_updating_heap:
			element = heap[element_idx]
			if element[1] > selected_element[1] or (element[1] == selected_element[1] and element[0] > selected_element[0]):
				heap[selected_son_idx] = heap[element_idx]
				heap[element_idx] = selected_element
				samples_idx_on_heap[selected_element[0]] = element_idx
				element_idx = selected_son_idx
			else:
				keep_updating_heap = False

	if heap_size >= 1:
		samples_idx_on_heap[heap[element_idx][0]] = element_idx

	return minElement, heap, heap_size, samples_idx_on_heap


class Clustering:

	def __init__(self, clustering_method, k, minPts, range_quantile=[0.25, 0.50, 0.75], single_gpu_for_clustering=True, gpu_id=0, version=""):

		self.clustering_method = clustering_method
		self.k = k
		self.minPts = minPts
		self.range_quantile = range_quantile
		self.single_gpu_for_clustering = single_gpu_for_clustering
		self.version = version
		self.gpu_id = gpu_id
		self.eps_values = []


	def fit_clustering(self, train_fvs):

		t0_clustering = time.time()
		if self.clustering_method == "LRR": 
			eps, D, I = defineEpsForDBSCAN(train_fvs, self.k, self.minPts, self.range_quantile, pipeline_iter, 
											single_gpu_for_clustering=self.single_gpu_for_clustering, v="R50_%s" % version)
			print(colored("Eps value for ResNet50: %.2f" % eps, "cyan"))
			pseudo_labels, core_points = GPUDBSCAN(D, I, self.k, minPts=self.minPts, eps=eps)
			non_outliers_idx = np.where(pseudo_labels != -1)[0]
			self.eps_values.append(eps)

		elif self.clustering_method == "FRR":

			#k1 = 30
			distances = compute_jaccard_distance(train_fvs, k1=self.k)
			distances = np.abs(distances)
			pseudo_labels = DBSCAN(eps=eps, min_samples=self.minPts, metric='precomputed', n_jobs=-1).fit_predict(distances)
			non_outliers_idx = np.where(pseudo_labels != -1)[0]

		elif self.clustering_method == "vanilla":

			distances = 1.0 - torch.mm(train_fvs, train_fvs.T)
			distances = np.abs(distances.numpy())
			pseudo_labels = DBSCAN(eps=eps, min_samples=self.minPts, metric='precomputed', n_jobs=-1).fit_predict(distances)
			non_outliers_idx = np.where(pseudo_labels != -1)[0]

		elif self.clustering_method == "PEACH":

			pseudo_labels = PEACH(train_fvs.numpy(), self.gpu_id, metric="cosine", batch_size = 4096, no_singleton=False, evt=False)
			all_pseudo_labels, freqs = np.unique(pseudo_labels, return_counts=True)

			if all_pseudo_labels[0] == -1:
				j = 1
			else:
				j = 0

			for lidx in range(len(all_pseudo_labels[j:])):
				if freqs[j:][lidx] == 1:
					pslabel =  all_pseudo_labels[j:][lidx]
					pslabel_idx = np.where(pseudo_labels == pslabel)[0]
					pseudo_labels[pslabel_idx] = -1						

			non_outliers_idx = np.where(pseudo_labels != -1)[0]

		else:
			print("Clustering method not implemented. Try 'LRR' or 'FRR'")
			exit()

		tf_clustering = time.time()
		dt_clustering = tf_clustering - t0_clustering
		print("Clustering time: {:2.2f} seconds".format(dt_clustering))

		return pseudo_labels, non_outliers_idx
