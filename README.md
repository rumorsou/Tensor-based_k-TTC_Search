# Tensor-based Triangle Connected k-Truss(k-TTC)
This repository contains the source code of the paper "Accelerating Triangle-Connected Truss Community Search Across Heterogeneous Hardware" .

## Overview
We provide source codes of tensor-based EquiTree index construction algorithms, including TETree and TETree-Basic, along with the search algorithm TETree-OPCC. All these algorithms are implemented by Python in the PyTorch framework.

## Experimental Environments
The operating system is Ubuntu 22.04, and development tools such as g++ 11.4, Python 3.11, PyTorch 2.2.2, torch-scatter 2.1.2, and CUDA 12.1 are installed to ensure that the test environment can fully satisfy all algorithms. 

## Datasets
The datasets are sourced from well-known platforms such as [SNAP (Stanford Network Analysis Platform)](https://snap.stanford.edu/data/) and [the Network Repository](https://networkrepository.com/index.php). Please ensure that there are no comments in the dataset file. The input dataset should be a text file where each line contains an edge in the format "u,v,k". Here, (u,v) represents an edge with u < v, and k denotes the trussness. Vertex IDs start from 0 and are ranked in increasing order.

## Running
1. Run TETree/TETree-Basic index construction algorithms using the following command (the `-f` parameter represents the dataset file name).

    for TETree:
  
      ```
   python ./TETree/TETree.py -f ./TETree/facebook.txt
      ```
    
    for TETree-Basic:
  
   ```
   python ./TETree/TETree-basic.py -f ./TETree/facebook.txt
   ```

    We also provide TETree-special-optimized, where triangle computation is specifically optimized for NVIDIA GPUs. See `./TETree/TETree-special-optimized/README_special.md` for details.

3. Run TETree-OPCC search algorithm using the following command (the `-f` parameter represents the dataset file name, the `-k` parameter represents "k"-TTC to be searched).

    for single vertex query (the `-v` parameter represents the vertex ID): 

     ```
     python TrussQuery.py -f ./facebook.txt -v 10 -k 4
     ```

    for random queries by count (the `-c` parameter represents the number of randomly generated query vertices):
   
     ```
   python TrussQuery.py -f ./facebook.txt -c 1000 -k 4
     ```














