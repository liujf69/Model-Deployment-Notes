import torch
import torchvision

if __name__ == "__main__":
    model = torchvision.models.resnet50()
    model.cuda()
    inputs = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, inputs, 'resnet50.onnx',opset_version=13)