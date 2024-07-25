一共有6个文件和一个编译setup.py文件

common_cuda_helper.hpp
pytorch_cpp_helper.hpp
pytorch_cuda_helper.hpp
这三个是辅助的固定文件，里面包含了需要使用的包，以及一些通用函数，静态变量

custom_cpu.cpp
这个文件是用于绑定pytorch和C++的，根据输入的类型来确定调用对应的设备函数

custom_cuda.cu
导入kernel文件
CUDA函数的实现，根据输入来确定需要多少线程，并调用kernel


custom_cuda_kernel.cuh
kernel函数的具体实现

setup.py
固定格式，只要将.cpp和.cu文件进行编译即可

要注意nvcc和对应的pytorch版本，不然无法编译


生成的结果是build文件，进入其下的lib.linux-x86_64-3.7 就有生成的.so文件可以直接在该路径下进行导入模型
import torch
import custom_cuda

a=torch.randn(1,3,4,4)
b=torch.randn(1,3,4,4) 
a=a.cuda()
b=b.cuda()
custom_cuda.foward(a,b)


如果有简单的语法错误就会出现编译问题，所以要逐句编译检查