**由CustomOp 创建一个kernel类进行计算**
*CustomOp：管理kernel的输入输出类型和个数，以及他的名字，共五组*

*kernel类：获取参数后，调用Compute计算，共三个*


onnxruntime/include/onnxruntime/core/session下面
   <onnxruntime_cxx_api.h>
   <onnxruntime_c_api.h>
   
-!wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
-!tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
-用os修改环境变量


 有时候会出现无法加载库文件，手动加载即可
   ***
   import torch
   torch.ops.load_library('/kaggle/working/onnxruntime-linux-x64-1.8.1/lib/libonnxruntime.so')
   torch.ops.load_library('/kaggle/working/onnxruntime-linux-x64-1.8.1/lib/libonnxruntime.so.1.8.1')
   ***
 
*使用就是*

   ''
   ort_custom_op_path = get_onnxruntime_op_path()
   session_options = rt.SessionOptions()
   if os.path.exists(ort_custom_op_path):
       session_options.register_custom_ops_library(ort_custom_op_path)
   ''
*之后就可以在sess中使用*
 
