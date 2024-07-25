#include "pytorch_cpp_helper.hpp"
void CustomForwardCUDAKernelLauncher(Tensor input ,Tensor output);

void CustomBackwardCUDAKernelLauncher(Tensor input ,Tensor output);


void custom_forward_cuda(Tensor input,Tensor output)
{
    CustomForwardCUDAKernelLauncher(input ,output);
}
void custom_backward_cuda(Tensor input,Tensor output)
{
    CustomBackwardCUDAKernelLauncher(input ,output);
}


void CustomForwardCPUKernelLauncher(Tensor input ,Tensor output){};

void CustomBackwardCPUKernelLauncher(Tensor input ,Tensor output){};

void custom_forward_cpu(Tensor input,Tensor output)
{
    CustomForwardCPUKernelLauncher(input ,output);
}
void custom_backward_cpu(Tensor input,Tensor output)
{
    CustomBackwardCPUKernelLauncher(input ,output);
}



void custom_foward(Tensor input,Tensor output)
{if (input.device().is_cuda()){
            CHECK_CUDA_INPUT(input);
         CHECK_CUDA_INPUT(output);
        custom_forward_cuda(input,output);
    }else{
        CHECK_CPU_INPUT(input);
        CHECK_CPU_INPUT(output);
        custom_forward_cpu(input,output);
}
}


void custom_backward(Tensor input,Tensor output)
{if (input.device().is_cuda()){
        CHECK_CUDA_INPUT(input);
         CHECK_CUDA_INPUT(output);
        custom_backward_cuda(input,output);
}else{
        CHECK_CPU_INPUT(input);
        CHECK_CPU_INPUT(output);
        custom_backward_cpu(input,output);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){

    m.def("foward",&custom_foward,"CUDA FORWARD",py::arg("input"),py::arg("output"));
    m.def("backward",&custom_backward,"CUDA FORWARD",py::arg("input"),py::arg("output"));

}
