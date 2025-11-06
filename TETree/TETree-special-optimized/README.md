# TETree-special-optimized

## Overview
Here is the source code of "TETree-special-optimized". which is specially optimized for NVIDIA GPUs. We implement additional optimizations for triangle computation, which constitutes a performance bottleneck in the general version.

We will subsequently organize and provide an additional interface to construct indices for datasets without precomputed trussness results.

## Environments
Ubuntu 22.04, Python 3.11, PyTorch 2.2.2, torch-scatter 2.1.2, and CUDA 12.1.

## Datasets
Please ensure that there are no comments in the dataset file, the starting vertex is "0" and generate their trussness using any k-truss decomposition application out there and write them in a file in the following format "u v k" where, (u,v) is an edge and u < v and k is the trussness.

We also provide a sample dataset named "facebook_truss_result.txt" which satisfies all requirements mentioned above.

## Running
1. Modify the absolute path of the sources parameter in "./hpu_extension/setup.py", then modify the path in the following command line and run to install the trusstensor library. Note that the extra_compile_args parameter in setup.py should be configured according to your experimental environment. For instance, when using NVIDIA A100 GPUs, you need to specify the corresponding compute capability by modifying the gencode flag to -gencode=arch=compute_80,code=sm_80.

2. Run the following command to install the extended tensor operator library.
   
  `python ./hpu_extension/setup.py install`

3. Prepare the dataset and build the index (take facebook_truss_result.txt for example).
   
  for TETree:
  
    `python TETree.py -f ./facebook_truss_result.txt`
    
  for TETree-Basic:

    `python TETree-basic.py -f ./facebook_truss_result.txt`
  
  './facebook_truss_result.txt' can be replaced by other datesets.


   



