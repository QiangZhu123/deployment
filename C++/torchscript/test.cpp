#include <torch/script.h>
#include <torch/custom_class.h>
#include <iostream>
#include <string>
#include <vector>
// ===================自定义函数（注意数据类型）=====================

int64_t funcadd(int64_t  a,int64_t  b){

    return a+b;
}

// my_ops::warp_perspective 就会表示成torch.ops.myops.addfunc      
static auto registry=torch::RegisterOperators("myops::addfunc",&funcadd);
