#ifndef CUSTOM_CUDA_KERNEL_CUH
#define CUSTOM_CUDA_KERNEL_CUH

template <typename T>
__global__ void custom_forward_cuda_kernel(const T* mask_data, T* buffer_data){
   int index = threadIdx.x+blockIdx.x*blockDim.x;
   


}
template<typename T>
__global__ void custom_backward_cuda_kernel(const T* input,T * output)
{
    int index = threadIdx.x+blockIdx.x*blockDim.x;
    
 
}

#endif