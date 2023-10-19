import torch
import torchvision
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn

if __name__ == "__main__":

    quant_modules.initialize() # 自动添加QDQ模块
    model = torchvision.models.resnet50()
    model.cuda()
    inputs = torch.randn(1, 3, 224, 224, device = 'cuda')
    quant_nn.TensorQuantizer.use_fb_fake_quant = True # 开启 FB 伪量化
    torch.onnx.export(model, inputs, 'quant_resnet50_.onnx', opset_version = 13)