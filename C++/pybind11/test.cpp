#include <torch/extension.h> 
#include <torch/torch.h>
#include <vector>
#include <iostream>

torch::Tensor functest(torch::Tensor a,torch::Tensor b)
{
    return torch::add(a,b);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &functest, "new forward");
}
