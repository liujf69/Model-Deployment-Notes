import os
import copy

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fx.graph_module import ObservedGraphModule

from dataloader import prepare_dataloader
from train_val import train_model, evaluate_model

# 量化模型
def quant_fx(model):
    # 使用Pytorch中的FX模式对模型进行量化
    model.eval()
    qconfig = get_default_qconfig("fbgemm")  # 默认是静态量化
    qconfig_dict = {
        "": qconfig,
    }
    model_to_quantize = copy.deepcopy(model)
    
    # 通过调用prepare_fx和convert_fx直接量化模型
    prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
    # print("prepared model: ", prepared_model) # 打印模型
    quantized_model = convert_fx(prepared_model)
    # print("quantized model: ", quantized_model) # 打印模型

    # 保存量化后的模型
    torch.save(quantized_model.state_dict(), "r18_quant.pth")

# 校准函数
def calib_quant_model(model, calib_dataloader):
    # 判断model一定是ObservedGraphModule，即一定是量化模型，而不是原始模型nn.module
    assert isinstance(
        model, ObservedGraphModule
    ), "model must be a perpared fx ObservedGraphModule."
    model.eval()
    with torch.inference_mode():
        for inputs, labels in calib_dataloader:
            model(inputs)
    print("calib done.")

# 比较校准前后的差异
def quant_calib_and_eval(model, test_loader):
    model.to(torch.device("cpu"))
    model.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        "": qconfig,
    }

    # 原始模型（未量化前的结果）
    print("model:")
    evaluate_model(model, test_loader)
    
    # 量化模型（未经过校准的结果）
    model2 = copy.deepcopy(model)
    model_prepared = prepare_fx(model2, qconfig_dict)
    model_int8 = convert_fx(model_prepared)
    print("Not calibration model_int8:")
    evaluate_model(model_int8, test_loader)

    # 通过原始模型转换为量化模型
    model3 = copy.deepcopy(model)
    model_prepared = prepare_fx(model3, qconfig_dict) # 将模型准备为量化模型，即插入观察节点
    calib_quant_model(model_prepared, test_loader)  # 使用数据对模型进行校准
    model_int8 = convert_fx(model_prepared) # 调用convert_fx将模型设置为量化模型
    torch.save(model_int8.state_dict(), "r18_quant_calib.pth") # 保存校准后的模型
    
    # 量化模型（已经过校准的结果）
    print("Do calibration model_int8:")
    evaluate_model(model_int8, test_loader)

if __name__ == "__main__":
    # 准备训练数据和测试数据
    train_loader, test_loader = prepare_dataloader()
    
    # 定义模型
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)
    
    # 训练模型（如果事先没有训练）
    if os.path.exists("r18_row.pth"): # 之前训练过就直接加载权重
        model.load_state_dict(torch.load("r18_row.pth", map_location="cpu"))
    else:
        train_model(model, train_loader, test_loader, torch.device("cuda"))
        print("train finished.")
        torch.save(model.state_dict(), "r18_row.pth")
        
    # 量化模型
    quant_fx(model)
    
    # 对比是否进行校准的影响
    quant_calib_and_eval(model, test_loader)