# TETree

We provide the source code of the paper "Accelerating Triangle-Connected Truss Community Search Across Heterogeneous Hardware" to construct EquiTree index here. This is a general-purpose version that can run on hardwares supporting PyTorch. For better performance, we have further optimized it for NVIDIA GPUs, and the optimized source code is available in "./TETree-special-optimized".

## Environment
Please ensure that your device can access PyTorch and torch_scatter.


'python TETree.py -f dataset_'
'python TETree-basic.py -f '
来调用构建算法


针对nvidia平台，我们做了特殊的改进，见TETree-special-optimized



