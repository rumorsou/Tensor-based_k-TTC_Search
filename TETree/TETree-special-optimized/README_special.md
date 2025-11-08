# TETree-special-optimized

## Overview
Here is the source code of "TETree-special-optimized", where triangle computation is specifically optimized for NVIDIA GPUs.

## Running
1. Modify the absolute path of the sources parameter in "./hpu_extension/setup.py", then modify the path in the following command line and run to install the trusstensor library. Note that the extra_compile_args parameter in setup.py should be configured according to your experimental environment. For instance, when using NVIDIA A100 GPUs, you need to specify the corresponding compute capability by modifying the gencode flag to -gencode=arch=compute_80,code=sm_80.
2. Modify the "./hpu_extension/setup.py" to configure your experimental environment. (1) replace lines 10-12 with your own path  (2) update the compile arguments in lines 

3. Run the following command to install the extended tensor operator library.

   `python ./hpu_extension/setup.py install`

4. Run TETree-special-optimized using the following command (note that '../facebook.txt' can be replaced by other datesets).
   
  
    python TETree.py -f ../facebook.txt
    
    or

    python TETree-basic.py -f ../facebook.txt
  



   














