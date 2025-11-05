修改TETree/TETree-optimized/hpu_extension/setup.py里的代码路径



执行python /home/featurize/work/TETree/TETree-optimized/hpu_extension/setup.py install 安装cuda加速包

特定于nvidia平台，用于加速三角形计算



安装pytorch

pip install  torch-scatter



准备好graph+truss文件，使用同样的方式运行

python xxx.py