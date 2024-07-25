pybind11和torchstript是两种extension方法，都是对pytorch的延伸

两个的不同点：
1 导入包不同

      torchstript：
      #include <torch/script.h>
      #include <torch/custom_class.h>

      pybind11：
      #include <torch/extension.h> 

2 延伸
  torchstript：
  
      static auto registry=torch::RegisterOperators("myops::addfunc",&funcadd);
      
  pybind11：
  
      PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        //python中名字   函数指针    函数注释
      m.def("forward", &functest, "new forward");

      //类的引入
      pybind11::class_<Name>(m,"pyname")
      .def(pybind11::init<int,int>())
      .def("get_a",&Name::get_a)
      .def("get_b",&Name::get_b);}
3 使用不同
    
    先Import torch
    都是要进入build/lib.linux-x86_64-3.7/路径下面，找到生成的.os文件
    torchstript：需要使用torch.ops.load_library方法加载文件，并使用torch.ops.myops.addfunc方法调用

    pybind11：
    直接导入函数即可