#include "custom_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

/*
void CustomForwardCUDAKernelLauncher(Tensor input ,Tensor output){
简化的函数调用
custom_forward_cuda_kernel<float><<<THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        input.data_ptr<float>(),output.data_ptr<float>());

};
*/
void CustomForwardCUDAKernelLauncher(Tensor input ,Tensor output){

    //计算需要的线程数量
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);
  
  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "custom_forward_cuda_kernel", [&] {
        custom_forward_cuda_kernel<scalar_t><<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(),output.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError()); 

};



void CustomBackwardCUDAKernelLauncher(Tensor input ,Tensor grad_output){

  int output_size = grad_output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);
  
  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "custom_backward_cuda_kernel", [&] {
        custom_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                grad_output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
  
};
