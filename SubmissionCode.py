#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import csv
def compute_ln_norm_distance(vector1, vector2, n):
    vector_len = len(vector1)
    diff_vector = []
    for i in range(0, vector_len):
      abs_diff = abs(vector1[i] - vector2[i])
      diff_vector.append(abs_diff ** n)
    ln_norm_distance = (sum(diff_vector))**(1.0/n)
    return ln_norm_distance

def find_k_nearest_neighbors(train_X, test_example, k, n_in_ln_norm_distance):
    indices_dist_pairs = []
    index= 0
    for train_elem_x in train_X:
      distance = compute_ln_norm_distance(train_elem_x, test_example,n_in_ln_norm_distance)
      indices_dist_pairs.append([index, distance])
      index += 1
    indices_dist_pairs.sort(key = lambda x: x[1])
    top_k_pairs = indices_dist_pairs[:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices

def classify_points_using_knn(train_X, train_Y, test_X, n_in_ln_norm_distance, k):
    test_Y = []
    for test_elem_x in test_X:
      top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k,n_in_ln_norm_distance)
      top_knn_labels = []
      for i in top_k_nn_indices:
        top_knn_labels.append(train_Y[i])
      most_frequent_label = max(set(top_knn_labels), key = top_knn_labels.count)
      test_Y.append(most_frequent_label)
    return test_Y

a=np.genfromtxt('train_X.csv',delimiter=',')
b=np.genfromtxt('train_Y.csv',delimiter=',')
a=np.delete(a,0,0)
co=np.shape(a)[0]
ro=np.shape(a)[1]
aw=np.empty(co, dtype=int)
test_Y = classify_points_using_knn(a, b, a, 3, 3)
for qw in range(0,len(test_Y)):
    aw[qw]=test_Y[qw]
aw=aw.reshape(co,1)
np.savetxt("predicted_test_Y.csv", aw, delimiter=",")


# In[ ]:




