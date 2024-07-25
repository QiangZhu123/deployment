from torch.autograd import Function
import test_cpp_forward,test_cpp_backward
'''
from ..utils import ext_loader
#导入mmcv._ext，保证'bbox_overlaps'存在
ext_module = ext_loader.load_ext('_ext', ['bbox_overlaps'])

'''
class TestFunction(Function):

    @staticmethod
    def forward(ctx, x, y,z,shape,mask,mode=True):
        ctx.shape=shape
        ctx.mask=mask
        ctx.mode=mode
        ctx.save_for_backward(x,y,z)
        output = x.new_zero()
        test_cpp_forward(x,y,z,shape,mask,output)

        return output,other

    @staticmethod
    def backward(ctx, gradOutput,gradother):
        shape=ctx.shape
        mask=ctx.mask
        x,y,z=ctx.saved_tensors
        gradX=x.new_zeros()
        gradY=y.new_zeros()
        gradZ=z.new_zeros()
        test_cpp_backward(gradOutput,gradX,gradY,gradZ)
        return gradX, gradY,None,None,None
    
myfunc=TestFunction.apply

class Test(torch.nn.Module):

    def __init__(self):
        super(Test, self).__init__()
        
    def forward(self, inputA, inputB):
        return myfunc(inputA,inputB,Z,shape,mask,mode=True)