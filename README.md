# Tensor-based Triangle Connected k-Truss(k-TTC)
This repository contains the source code of the paper "A Tensor-based Framework for Efficient Triangle-Connected Truss Community Search" .

## Overview
We provide source codes of tensor-based algorithms, including truss decomposition, EquiTruss/EquiTree index construction, k-TTC search and index maintenance. All these algorithms are implemented bu Python in the PyTorch framework.

## Experimental Environments
The operating system is Ubuntu 22.04, and development tools such
as g++ 11.4, Python 3.11, PyTorch 2.2.2, torch-scatter 2.1.2, and
CUDA 12.1 are installed to ensure that the test environment can
fully satisfy all algorithms. 

## Datasets
The datasets are sourced from well-known platforms such as
[SNAP (Stanford Network Analysis Platform)](https://snap.stanford.edu/data/) and [the Network Repository](https://networkrepository.com/index.php). Please ensure that there are no comments in the dataset file, the starting vertex is "0", and each line stores the source vertex and the destination vertex u and v (u < v) of an edge.

## Running
Modify the absolute path of the graph dataset in the `run` function of the `main` entry in the `equitruss.py` and `equitree.py` files, then execute `python xxx.py` to complete the index construction. For index search and index maintenance, please uncomment the corresponding sections in the `run` function.

If you have any questions, you can send an email to junchaoma@whu.edu.cn or xin_yan@whu.edu.cn
