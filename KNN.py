import numpy as np
import sys
import os
import aghasher
import scipy.sparse
import math
import multiprocessing as mp
import itertools
#import CPFcluster.utils as utils
import utils

def get_knn(X, k, reps = 50, num_hashbits = 12, blocksz = 100, n_core = 1):
  if X.shape[0] < 1000:
    edge_arr = np.empty((X.shape[0], k))
    edge_arr[:] = np.nan
    w_arr = np.ones((X.shape[0], k))*np.inf
    PDmat = distance_matrix(X)
    for i in range(len(PDmat)):
      w_arr[i, 0:k] = np.sort(PDmat[i,:])[1:k+1]
      edge_arr[i, 0:k] = np.argsort(PDmat[i,:])[1:k+1]
    
    return edge_arr, w_arr
  else: 
    edge_arr = np.empty(shape = (X.shape[0], k))
    edge_arr[:] = np.nan
    w_arr = np.ones((X.shape[0], k))*np.inf
    for rep in range(reps):
      edge_arr, w_arr = basic_ann_lsh(X, k, num_hashbits, blocksz, n_core, edge_arr, w_arr)
    
    edge_arr, w_arr = refine(X, edge_arr, w_arr)
    return edge_arr, w_arr

def basic_ann_lsh(X, k, m, blocksz, n_core, edge_arr, w_arr):
  agh = None
  while agh is None:
    try:
      anchoridx = np.random.choice(X.shape[0], size = 25, replace = False)
      anchor = X[anchoridx, :]
      agh, H_train = aghasher.AnchorGraphHasher.train(X, anchor, m)
    except:
      pass
  
  Y = agh.hash(X)
  Y = Y.astype(int)
  w = np.random.rand(m, 1)
  p = Y.dot(w)
  p = np.ravel(p)
  pid = np.argsort(p)
  l = ([X[block,: ], block, k, blocksz, edge_arr[block,:], w_arr[block,:]] for block in list(chunks(pid, blocksz)))
  try:
    pool = mp.Pool(processes=n_core)              
    result = []
    N = max(math.floor(X.shape[0]/(100*blocksz)), 20)
    while True:
      g2 = pool.map(func_star, itertools.islice(l, N))
      if g2:
        result.append(g2)
      else:
        break
    pool.terminate()
  except Exception as e:
    print("POOL ERROR")
    pool.close()
    pool.terminate()
  
  res_flat = [res_pair for res_list in result for res_pair in res_list]
  repedge_arr, repw_arr = zip(*res_flat)
  repedge_arr = np.vstack(repedge_arr)
  repw_arr = np.vstack(repw_arr)
  repedge_arr = repedge_arr[np.argsort(pid),:]
  repw_arr = repw_arr[np.argsort(pid),:]
  return repedge_arr, repw_arr

def refine(X, edge_arr, w_arr):
  edge_arr = edge_arr.astype(int)
  ref_edge_arr = edge_arr
  ref_w_arr = w_arr
  k = edge_arr.shape[1]
  for j in range(k):
    ref_edge_arr = np.hstack((ref_edge_arr, edge_arr[edge_arr[:,j], :]))
    unidx = list(map(unique_idx, ref_edge_arr))
    refined = (np.setdiff1d( ref_edge_arr[row, unidx[row]], row) for row in range(len(unidx)))
    i = 0
    for i_edge_list in refined:
      iw = np.sqrt(np.square(X[i,:] - X[i_edge_list,:]).sum(1))
      ref_w_arr[i,:] = np.sort(iw)[0:k]
      ref_edge_arr[i,0:k] = i_edge_list[np.argsort(iw)[0:k]]
      i += 1
    
    ref_edge_arr = ref_edge_arr[:,0:k]
  
  return ref_edge_arr, ref_w_arr  


def chunks(lst, n):
  for i in range(0, len(lst), n):
    yield lst[i:i+n]

def distance_matrix(A):
  return np.sqrt(np.square(A[np.newaxis,:,:]-A[:,np.newaxis,:]).sum(2))

def unique_idx(row):
  return np.unique(row, return_index = True)[1]

def brute_force_knn_par(A, idx, k, blocksz, edge_mat, w_mat):
  mat = distance_matrix(A)
  dismat = np.hstack((w_mat, mat))
  idxmat = np.hstack((edge_mat, np.tile(idx, (len(idx), 1))   ))
  outidx = np.ones(shape = (len(idx), k))*np.inf
  outdis = np.ones(shape = (len(idx), k))*np.inf
  for i in range(len(mat)):
    vals, unidx = np.unique(idxmat[i,:], return_index = True)
    unidx = list(unidx)
    if i+k not in unidx:
      print(idx)
      print(i)
      sys.stdout.flush()
    unidx.remove([i+k])
    outdis[i, 0:min(len(unidx)-1, k)] = np.sort(dismat[i,unidx])[0:min(len(unidx) - 1, k)]
    min_k_indices = [unidx[i] for i in np.argsort(dismat[i,unidx])[0:min(len(unidx) -1, k)]]
    outidx[i, 0:min(len(unidx)-1, k)] = idxmat[i,min_k_indices]
    
  return [outidx, outdis]

def func_star(a_b):
  try:
    return brute_force_knn_par(*a_b)
  except Exception as e:
    raise Exception(e)

