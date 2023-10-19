import torch
from pytorch_quantization import tensor_quant


if __name__ == "__main__":
    torch.manual_seed(123456)
    x = torch.rand(10)
    fake_x = tensor_quant.fake_tensor_quant(x, x.abs().max()) # 传入输入数据及其最大的绝对值
    print(x)
    print(fake_x)

    # tensor([0.5043, 0.8178, 0.4798, 0.9201, 0.6819, 0.6900, 0.6925, 0.3804, 0.4479, 0.4954])
    # tensor([0.5071, 0.8187, 0.4782, 0.9201, 0.6810, 0.6883, 0.6955, 0.3840, 0.4492, 0.4927])