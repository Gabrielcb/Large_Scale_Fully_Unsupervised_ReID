import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

LDTYPE = np.int64
ctypedef np.int64_t LDTYPE_t

FTYPE = np.float32
ctypedef np.float32_t FTYPE_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def ReRankingV2(np.ndarray[FTYPE_t, ndim=2] D, np.ndarray[DTYPE_t, ndim=2] I):

	print("Performing Re-Ranking...")
	cdef np.ndarray[FTYPE_t, ndim=2] RD = np.zeros([D.shape[0], D.shape[1]], dtype=FTYPE)
	cdef np.ndarray[DTYPE_t, ndim=2] RI = np.zeros([I.shape[0], I.shape[1]], dtype=DTYPE)

	cdef np.ndarray[DTYPE_t, ndim=1] x_verification = np.zeros(D.shape[1], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] y_verification = np.zeros(D.shape[1], dtype=DTYPE)

	cdef int n = D.shape[0]
	cdef int k = D.shape[1]

	cdef FTYPE_t J, min_sum, max_sum
	cdef int sample_idx, neighbor_index, i, j, idx, fi, gi
	cdef np.ndarray[DTYPE_t, ndim=1] sample_neighbors
	cdef np.ndarray[DTYPE_t, ndim=1] neigh_neighbors
	#cdef np.ndarray[DTYPE_t, ndim=1] elements
	#cdef np.ndarray[DTYPE_t, ndim=1] idx_on_first
	#cdef np.ndarray[DTYPE_t, ndim=1] idx_on_second
	#cdef np.ndarray[DTYPE_t, ndim=1] idx_only_on_first
	#cdef np.ndarray[DTYPE_t, ndim=1] idx_only_on_second

	cdef np.ndarray[FTYPE_t, ndim=2] expD = np.exp(-D)
	cdef np.ndarray[LDTYPE_t, ndim=2] sorted_idx_by_row = np.zeros([I.shape[0], I.shape[1]], dtype=LDTYPE) 

	print(n,k)

	for sample_idx in range(n):
		sample_neighbors = I[sample_idx]
		for j in range(k):
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

					#elements, idx_on_first, idx_on_second = np.intersect1d(sample_neighbors,neigh_neighbors, return_indices=True)
					#idx_only_on_first = np.setdiff1d(np.arange(k), idx_on_first)
					#idx_only_on_second = np.setdiff1d(np.arange(k), idx_on_second)

					x_verification = np.zeros(k, dtype=DTYPE)
					y_verification = np.zeros(k, dtype=DTYPE)

					for fi in range(k):
						for gi in range(k):
							if sample_neighbors[fi] == neigh_neighbors[gi]:
								x_verification[fi] = 1
								y_verification[gi] = 1

								min_sum += min(expD[sample_idx, fi], expD[neighbor_index, gi])
								max_sum += max(expD[sample_idx, fi], expD[neighbor_index, gi])

								break

					for fi in range(k):
						if x_verification[fi] == 0:
							max_sum += expD[sample_idx, fi]

					for gi in range(k):
						if y_verification[gi] == 0:
							max_sum += expD[neighbor_index, gi]


					#print("Elements in common:", elements)
					#print("First element idx:", idx_on_first)
					#print("Second element idx:", idx_on_second)	
					#print("Idx only on first:", idx_only_on_first)
					#print("Idx only on second:", idx_only_on_second)

					#min_sum += np.minimum(expD[sample_idx, idx_on_first], expD[neighbor_index, idx_on_second]).sum()
					#max_sum += np.maximum(expD[sample_idx, idx_on_first], expD[neighbor_index, idx_on_second]).sum()

					#print("Min and Max sum:", min_sum, max_sum)

					#max_sum += expD[sample_idx, idx_only_on_first].sum() + expD[neighbor_index, idx_only_on_second].sum()

					#print("Max sum:", max_sum)

					J = 1 - (min_sum/max_sum)

					#print(D[sample_idx, j], J)
					RD[sample_idx, j] = J
					#print("Jaccard distance:", J)
					idx = np.where(I[neighbor_index] == sample_idx)[0][0]

					#print("Localition of sample in mutual neighborhood", idx)
					#if len(idx) > 0:
					RD[neighbor_index, idx] = J

				else:
					RD[sample_idx, j] = 1.0

			else:

				if not sample_idx in I[neighbor_index]:
					RD[sample_idx, j] = 1.0


		#exit()

	sorted_idx_by_row = np.argsort(RD, axis=1)
	RD = np.array([RD[i,sorted_idx_by_row[i]] for i in range(RD.shape[0])], dtype=FTYPE)
	RI = np.array([I[i,sorted_idx_by_row[i]] for i in range(I.shape[0])], dtype=DTYPE)
	#NSA
	return RD, RI