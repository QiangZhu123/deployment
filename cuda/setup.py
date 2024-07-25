import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#import os
#import glob
# 头文件目录
#include_dirs = os.path.dirname(os.path.abspath(__file__))
# 源代码目录
#source_cpu = glob.glob(os.path.join(include_dirs, 'cpu', '*.cpp'))


#模型名字是 import custom_cuda,只要给定.cpp和.cu文件即可
setup(name='custom',
    ext_modules=[
        CUDAExtension('custom_cuda', [
         'custom.cpp',
         'custom_cuda.cu'
        ])
    ],
    extra_compile_args={'cxx': ['gcc']},
    cmdclass={
        'build_ext': BuildExtension
    })