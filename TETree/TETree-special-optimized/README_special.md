# TETree-special-optimized

## Overview
Here is the source code of "TETree-special-optimized", where triangle computation is specifically optimized for NVIDIA GPUs.

## Running
1. Modify "./hpu_extension/setup.py" to configure your experimental environment:
   
   (1) replace lines 10-12 with your own path.
   
   (2) update the compile arguments in lines 22-23.

2. Run the following command to install the extended tensor operator library.

   `python ./hpu_extension/setup.py install`

3. Run TETree-special-optimized using the following command (note that '../facebook.txt' can be replaced by other datesets).
   
  
    `python TETree.py -f ../facebook.txt`
    
    or

    `python TETree-basic.py -f ../facebook.txt`
  



   


















