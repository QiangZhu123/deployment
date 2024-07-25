from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
include_dirs = os.path.dirname(os.path.abspath(__file__))
setup(name='myop_file',
      ext_modules=[cpp_extension.CppExtension('myadd',
                       ['test.cpp'],
                  include_dirs=[include_dirs])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
