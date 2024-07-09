import torch
import numpy as np
from termcolor import colored

def BatchCenterLoss(batch_fvs, batch_labels, centers, centers_labels, tau=0.1, gpu_index=0):

	# Calculating Similarity
	S = torch.matmul(batch_fvs, centers.T)
	#centers_labels = torch.Tensor(centers_labels)
	
	# Calcuating Loss
	batch_loss = torch.tensor(0.0).cuda(gpu_index)

	for si in range(batch_fvs.shape[0]):
		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]

		#print(colored("###====== Jesus ======###", "blue"))
		#print(fvs_similarities)
		
		# Proxy Loss
		positive_similarity = fvs_similarities[centers_labels == pseudo_label][0]
		#print(positive_similarity)

		pos_sim = torch.exp(positive_similarity/tau)
		all_sim = torch.exp(fvs_similarities/tau).sum()
		batch_loss += -torch.log(pos_sim/all_sim)
		#print(-torch.log(pos_sim/all_sim))

	batch_loss = batch_loss/batch_fvs.shape[0]	

	return batch_loss

def BatchSoftmaxTripletLoss(batch_fvs, batch_labels, batch_pids, tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	corrects = 0
	total_number_triplets = 0

	for si in np.arange(S.shape[0]):

		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		true_label = batch_pids[si] # Only for reference! It is NOT used on optmization!!

		positive_similarities = fvs_similarities[batch_labels == pseudo_label]
		negative_similarities = fvs_similarities[batch_labels != pseudo_label]

		#print(positive_similarities.shape, negative_similarities.shape)

		p, pos_idx = torch.topk(positive_similarities, k=1, largest=False)
		q, neg_idx = torch.topk(negative_similarities, k=1, largest=True)

		#print(p, pos_idx, q, neg_idx)

		p = torch.exp(p[0]/tau)
		q = torch.exp(q[0]/tau)

		sample_loss = -torch.log(p/(p+q))
		batch_loss += sample_loss

		pos_pid = batch_pids[batch_labels == pseudo_label][pos_idx[0]]
		neg_pid = batch_pids[batch_labels != pseudo_label][neg_idx[0]]

		#print(true_label, pos_pid, neg_pid)

		if (true_label == pos_pid) and (true_label != neg_pid):
			corrects += 1
		total_number_triplets += 1

	batch_loss = batch_loss/S.shape[0]
	return batch_loss, corrects, total_number_triplets


def BatchSoftmaxAllTripletLoss(batch_fvs, batch_labels, batch_pids, tau=0.1, gpu_index=0):
	  
	
	S = torch.mm(batch_fvs, batch_fvs.T)
	S_exp = torch.exp(S/tau)

	batch_labels = np.array([batch_labels])
	nb = batch_fvs.shape[0]
	
	batch_labels_expanded = np.repeat(batch_labels, nb, axis=0)
	batch_labels_expanded_transposed = np.repeat(batch_labels, nb, axis=0).T
	
	pos_mask = torch.Tensor(batch_labels_expanded == batch_labels_expanded_transposed).int().cuda(gpu_index)
	neg_mask = 1 - pos_mask

	pos_sim = S_exp*pos_mask
	neg_sim = S_exp*neg_mask

	neg_sum = torch.sum(neg_sim, dim=1, keepdim=True)
	relative_pos = -torch.log(S_exp/(S_exp+neg_sum))*pos_mask

	#pos_weights = pos_dist/torch.sum(pos_dist, dim=1, keepdim=True)
	#neg_weights = neg_dist/torch.sum(neg_dist, dim=1, keepdim=True)	

	batch_loss = torch.mean(torch.sum(relative_pos, dim=1)/torch.sum(pos_mask, dim=1))
	return batch_loss


def BatchCenterLossWithOutliers(batch_fvs, batch_labels, centers, centers_labels, tau=0.1, gpu_index=0):

	# Calculating Similarity
	S = torch.matmul(batch_fvs, centers.T)
	#centers_labels = torch.Tensor(centers_labels)
	
	# Calcuating Loss
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	Nbi = 0

	for si in range(batch_fvs.shape[0]):
		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		if pseudo_label != '-1.0_-1.0_-1.0':
			#print(colored("###====== Jesus ======###", "blue"))
			#print(fvs_similarities)
			
			# Proxy Loss
			positive_similarity = fvs_similarities[centers_labels == pseudo_label][0]
			#print(positive_similarity)

			pos_sim = torch.exp(positive_similarity/tau)
			all_sim = torch.exp(fvs_similarities/tau).sum()
			batch_loss += -torch.log(pos_sim/all_sim)
			#print(-torch.log(pos_sim/all_sim))
			Nbi += 1

	batch_loss = batch_loss/Nbi
	return batch_loss

def BatchSoftmaxTripletLossWithOutliers(batch_fvs, batch_labels, batch_pids, tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	corrects = 0
	total_number_triplets = 0
	Nbi = 0

	for si in np.arange(S.shape[0]):

		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		true_label = batch_pids[si] # Only for reference! It is NOT used on optmization!!

		if pseudo_label != '-1.0_-1.0_-1.0':

			positive_similarities = fvs_similarities[batch_labels == pseudo_label]
			negative_similarities = fvs_similarities[batch_labels != pseudo_label]

			#print(positive_similarities.shape, negative_similarities.shape)

			p, pos_idx = torch.topk(positive_similarities, k=1, largest=False)
			q, neg_idx = torch.topk(negative_similarities, k=1, largest=True)

			#print(p, pos_idx, q, neg_idx)

			p = torch.exp(p[0]/tau)
			q = torch.exp(q[0]/tau)

			sample_loss = -torch.log(p/(p+q))
			batch_loss += sample_loss

			pos_pid = batch_pids[batch_labels == pseudo_label][pos_idx[0]]
			neg_pid = batch_pids[batch_labels != pseudo_label][neg_idx[0]]

			#print(true_label, pos_pid, neg_pid)

			if (true_label == pos_pid) and (true_label != neg_pid):
				corrects += 1
			total_number_triplets += 1

			Nbi += 1

	batch_loss = batch_loss/Nbi
	return batch_loss, corrects, total_number_triplets

def BatchWeightedSoftmaxAllTripletLoss(batch_fvs, batch_labels, gpu_index=0):
  
	# Calculating Distance
	batch_labels = np.array([batch_labels])
	Dist = 1.0 - (torch.matmul(batch_fvs, batch_fvs.T) + 1.0)/2.0
	
	nb = batch_fvs.shape[0]
	#batch_labels_expanded = batch_labels.repeat(nb,1)
	#batch_labels_expanded_transposed = batch_labels.repeat(nb,1).T

	batch_labels_expanded = np.repeat(batch_labels, nb, axis=0)
	batch_labels_expanded_transposed = np.repeat(batch_labels, nb, axis=0).T

	pos_mask = torch.Tensor(batch_labels_expanded == batch_labels_expanded_transposed).int().cuda(gpu_index)
	neg_mask = 1 - pos_mask

	pos_dist = torch.exp(Dist)*pos_mask
	neg_dist = torch.exp(-Dist)*neg_mask

	pos_weights = pos_dist/torch.sum(pos_dist, dim=1, keepdim=True)
	neg_weights = neg_dist/torch.sum(neg_dist, dim=1, keepdim=True)	

	pos_loss = torch.sum(pos_weights*Dist, dim=1)
	neg_loss = torch.sum(neg_weights*Dist, dim=1)

	batch_loss = torch.mean(torch.log(1 + torch.exp(pos_loss - neg_loss)))
	return batch_loss


def BatchInstanceLossOutliers(batch_fvs, batch_labels, batch_pids, gpu_index=0):
  
	batch_fvs_outliers = batch_fvs[batch_labels == '-1.0_-1.0_-1.0']
	outlier_pids = batch_pids[batch_labels == '-1.0_-1.0_-1.0']
	corret_rates = np.sum(outlier_pids[::2] == outlier_pids[1::2])/outlier_pids[::2].shape[0]
	
	S = torch.mm(batch_fvs_outliers[::2], batch_fvs_outliers[1::2].T)
	batch_loss = torch.trace(1.0 - S)/S.shape[0]
	
	return batch_loss

def BatchMedianSoftmaxTripletLoss(batch_fvs, batch_labels, batch_pids, tau=0.1, gpu_index=0):
  
	S = torch.mm(batch_fvs, batch_fvs.T)
	batch_loss = torch.tensor(0.0).cuda(gpu_index)
	corrects = 0
	total_number_triplets = 0

	for si in np.arange(S.shape[0]):

		fvs_similarities = S[si]
		pseudo_label = batch_labels[si]
		true_label = batch_pids[si] # Only for reference! It is NOT used on optmization!!

		positive_similarities = fvs_similarities[batch_labels == pseudo_label]
		negative_similarities = fvs_similarities[batch_labels != pseudo_label]

		#print(positive_similarities.shape, negative_similarities.shape)


		#p, pos_idx = torch.topk(positive_similarities, k=1, largest=False)
		#q, neg_idx = torch.topk(negative_similarities, k=1, largest=True)

		p, pos_idx = torch.median(positive_similarities, dim=0)
		q, neg_idx = torch.median(negative_similarities, dim=0)

		#print(p, pos_idx, q, neg_idx)

		#exit()
		p = torch.exp(p/tau)
		q = torch.exp(q/tau)

		sample_loss = -torch.log(p/(p+q))
		batch_loss += sample_loss

		pos_pid = batch_pids[batch_labels == pseudo_label][pos_idx]
		neg_pid = batch_pids[batch_labels != pseudo_label][neg_idx]

		#print(true_label, pos_pid, neg_pid)

		if (true_label == pos_pid) and (true_label != neg_pid):
			corrects += 1
		total_number_triplets += 1

	loss = batch_loss/S.shape[0]
	return loss, corrects, total_number_triplets