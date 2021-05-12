# FastKNN
An implementation of the Fast kNN method introduced by 'Fast kNN Graph Construction with Locality Sensitive Hashing' by Zhang et. al. (2013).

For *X* an *n x d* numpy array, the *k* nearest neighbors are found using:
```python
edge_arr, w_arr = KNN.get_knn(X, k, reps = 50, num_hashbits = 12, blocksz = 100, n_core = 1)
```
Here, `edge_arr` and `w_arr` are *n x k* numpy arrays containing the index of the *k* nearest neighbors and the distances between the corresponding instances respectively. 

All questions and comments on the code are greatly appreciated!
