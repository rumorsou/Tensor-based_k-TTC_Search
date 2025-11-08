# Tensor-based Triangle Connected k-Truss(k-TTC)
This repository contains the source code of the paper "Accelerating Triangle-Connected Truss Community Search Across Heterogeneous Hardware" .

## Overview
We provide source codes of tensor-based EquiTree index construction algorithms, including TETree and TETree-Basic. All these algorithms are implemented by Python in the PyTorch framework.

## Experimental Environments
The operating system is Ubuntu 22.04, and development tools such as g++ 11.4, Python 3.11, PyTorch 2.2.2, torch-scatter 2.1.2, and CUDA 12.1 are installed to ensure that the test environment can fully satisfy all algorithms. 

## Datasets
The datasets are sourced from well-known platforms such as [SNAP (Stanford Network Analysis Platform)](https://snap.stanford.edu/data/) and [the Network Repository](https://networkrepository.com/index.php). Please ensure that there are no comments in the dataset file, the starting vertex is "0" and generate their trussness using any k-truss decomposition application out there and write them in a file in the following format "u,v,k" where, (u,v) is an edge and u < v and k is the trussness.

## Running
We provide the source code: TETree and TETree-Basic, which are general versions that can run on any hardware supporting PyTorch.

TETree and TETree-special-optimized. TETree is a general version that can run on any hardware supporting PyTorch, while TETree-special-optimized is further optimized for NVIDIA GPUs. Please refer to the corresponding readme.md for instructions on running the respective algorithms.






