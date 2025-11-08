# TETree-special-optimized

## Overview
Here is the source code of "TETree-special-optimized". which is specially optimized for NVIDIA GPUs. We implement additional optimizations for triangle computation, which constitutes a performance bottleneck in the general version.

## Running
1. Modify the absolute path of the sources parameter in "./hpu_extension/setup.py", then modify the path in the following command line and run to install the trusstensor library. Note that the extra_compile_args parameter in setup.py should be configured according to your experimental environment. For instance, when using NVIDIA A100 GPUs, you need to specify the corresponding compute capability by modifying the gencode flag to -gencode=arch=compute_80,code=sm_80.

2. Run the following command to install the extended tensor operator library.

   `python ./hpu_extension/setup.py install`

4. Prepare the dataset and build the index (take facebook_truss_result.txt for example).
   
  for TETree:
  
    python TETree.py -f ../facebook_truss_result.txt
    
  for TETree-Basic:

    python TETree-basic.py -f ../facebook_truss_result.txt
  
  '../facebook_truss_result.txt' can be replaced by other datesets.


   










