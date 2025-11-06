# TETree

We provide the source code of the paper "Accelerating Triangle-Connected Truss Community Search Across Heterogeneous Hardware" to construct EquiTree index here. 

## Overview
This is a general-purpose version that can run on hardwares supporting PyTorch. For better performance, we have further optimized it for NVIDIA GPUs, and the optimized source code is available in "./TETree-special-optimized".

We will subsequently organize and provide the complete source code for the search and maintenance algorithms, along with additional interfaces to construct indices for datasets without precomputed trussness results.

## Environments
Please ensure that your device can access PyTorch and torch_scatter.

## Datasets
Please ensure that there are no comments in the dataset file, the starting vertex is "0" and generate their trussness using any k-truss decomposition application out there and write them in a file in the following format "u v k" where, (u,v) is an edge and u < v and k is the trussness.

We also provide a sample dataset named "facebook_truss_result.txt" which satisfies all requirements mentioned above.

## Running
for TETree:<br>
  `python TETree.py -f ./facebook_truss_result.txt`<br>
for TETree-Basic:<br>
  `python TETree-basic.py -f ./facebook_truss_result.txt`<br>

'./facebook_truss_result.txt' can be replaced by other datesets.








