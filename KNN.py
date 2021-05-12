import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import sharedmem
import itertools
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import scipy.stats
import aghasher
import math
import sys
import os
#from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances

def get_knn(X, k, reps = 50, num_hashbits = 12, blocksz = 100, n_core = 1):
  X = np.array(X).astype(float)
  edge_arr = np.empty(shape = (X.shape[0], k))
  edge_arr[:] = np.nan
  w_arr = np.empty(shape = (X.shape[0], k))
  w_arr[:] = np.nan
  if X.shape[0] < 1000:
    PDmat = distance_matrix(X)
    w_arr = np.array(list(map(np.sort, PDmat)))[:, range(k)]
    edge_arr = np.array(list(map(np.argsort, PDmat))).astype(int)[:,range(k)]
    return edge_arr, w_arr
  else:
    nrep = 25
    for rep in tqdm(range(nrep), "Computing Nearest Neighbors:"):
      edge_arr, w_arr = basic_ann_lsh(X, k, num_hashbits, blocksz, n_core, edge_arr, w_arr)
    
    narows = [x for x in range(edge_arr.shape[0]) if np.inf in edge_arr[x, :]]
    for x in narows:
      na_dists = np.sqrt(np.square(X[x,:] - X).sum(1))
      w_arr[x,:] = np.sort(na_dists)[range(1, edge_arr.shape[1] + 1)]
      edge_arr[x, :] = np.argsort(na_dists)[range(1, edge_arr.shape[1] + 1)]
    
    edge_arr, w_arr = refine(X, edge_arr, w_arr)
    return edge_arr, w_arr

#Functions Used
def basic_ann_lsh(X, k, m, blocksz, n_core, edge_arr, w_arr):
  agh = None
  while agh is None:
    try:
      anchoridx = np.random.choice(X.shape[0], size = 25, replace = False)
      anchor = X[anchoridx, :]
      agh, H_train = aghasher.AnchorGraphHasher.train(X, anchor, m)
    except:
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
    #result = [[]]*math.ceil(len(list(chunks(pid, blocksz)))/n_core)
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
  n = edge_arr.shape[0]
  k = edge_arr.shape[1]
  edge_arr = edge_arr.astype(int)
  in_edge_arr = edge_arr.astype(int)
  chunks = chunks(range(n), 20000)
  for j in range(k):
    for chunk in chunks:
      ref_edge_arr = in_edge_arr[in_edge_arr[chunk,j], :]
      stack_edge_arr = np.hstack((edge_arr[chunk,:], ref_edge_arr))
      X_NNchunk = X[ref_edge_arr, :]
      ref_w_arr = np.sqrt(np.square(X[chunk,np.newaxis,:] - X_NNchunk).sum(2))
      stack_w_arr = np.hstack((w_arr[chunk,:], ref_w_arr))
      unidx = [unique_idx(stack_edge_arr[i,:]) for i in range(len(chunk))]
      unidx = [unidx[i][stack_edge_arr[i, unidx[i]] != chunk[i]] for i in range(len(chunk))]
      sortidx = [np.argsort(stack_w_arr[i, unidx[i]])[range(k)] for i in range(len(chunk))]
      edge_arr[chunk,:] = np.stack([stack_edge_arr[i, unidx[i]][sortidx[i]] for i in range(len(chunk))])
      w_arr[chunk,:] = np.stack([stack_w_arr[i, unidx[i]][sortidx[i]] for i in range(len(chunk))])
  
  return edge_arr, w_arr

def chunks(lst, n):
  for i in range(0, len(lst), n):
    yield lst[i:i+n]

def skip_diag_strided(A):
  m = A.shape[0]
  strided = np.lib.stride_tricks.as_strided
  s0,s1 = A.strides
  return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)

def distance_matrix(A):
  M = np.sqrt(np.square(A[np.newaxis,:,:]-A[:,np.newaxis,:]).sum(2))
  M = skip_diag_strided(M)
  return M

def unique_idx(row):
  return np.unique(row, return_index = True)[1]

def idx_unique(a):
  weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
  b = a + weight[:, np.newaxis]
  u, ind = np.unique(b, return_index=True)
  b = np.zeros_like(a)
  np.put(b, ind, np.ones(shape = a.shape).flat[ind])
  return b.astype(bool)

def brute_force_knn_par(A, idx, k, blocksz, edge_mat, w_mat):
  mat = distance_matrix(A)
  dismat = np.hstack((w_mat, mat))
  tile_idx = skip_diag_strided(np.tile(idx, (len(idx), 1)))
  idxmat = np.hstack((edge_mat,  tile_idx ))
  sortdis = np.argsort(dismat, axis = -1)
  dismat = np.sort(dismat, axis = -1)
  idxmat = idxmat[np.arange(dismat.shape[0])[:,None], sortdis]
  unidx = idx_unique(idxmat)
  outdis = np.array([dismat[i, unidx[i]][0:k] for i in range(dismat.shape[0])])
  outidx = np.array([idxmat[i, unidx[i]][0:k] for i in range(dismat.shape[0])])
  return [outidx, outdis]
