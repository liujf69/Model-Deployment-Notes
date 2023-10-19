import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor

class QuantMultiAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histgoram"))
    
    def forward(self, x, y, z):
        return self._input_quantizer(x) + self._input_quantizer(y) + self._input_quantizer(z)

if __name__ == "__main__":
    model = QuantMultiAdd()
    model.cuda()
    input_a = torch.randn(1, 3, 224, 224, device='cuda')
    input_b = torch.randn(1, 3, 224, 224, device='cuda')
    input_c = torch.randn(1, 3, 224, 224, device='cuda')
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    torch.onnx.export(model, (input_a, input_b, input_c), 'quantMultiAdd.onnx', opset_version = 13)
