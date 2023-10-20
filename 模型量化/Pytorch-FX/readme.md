#  Torch-FX量化
Pytorch在torch.quantization.quantize_fx中提供了两个API，即prepare_fx和convert_fx。  
prepare_fx的作用是准备量化，其在输入模型里按照设定的规则qconfig_dict来插入观察节点，进行的工作包括:
```
1. 将nn.Module转换为GraphModule
2. 合并算子，例如将Conv、BN和Relu算子进行合并（通过打印模型可以查看合并的算子）
3. 在Conv和Linear等OP前后插入Observer, 用于观测激活值Feature map的特征（权重的最大最小值），计算scale和zero_point
```
convert_fx的作用是根据scale和zero_point来将模型进行量化。

# 校准模型
在对原始模型model调用prepare_fx()后得到prepare_model，一般需要对模型进行校准，校准后再调用convert_fx()进行模型的量化。

# 代码实例
## 主函数
```python
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
```

## prepare_dataloader函数
```python
# 准备训练数据和测试数据
def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform
    )
    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
    )
    return train_loader, test_loader
```

## 训练和测试函数
```python
# 训练模型，用于后面的量化
def train_model(model, train_loader, test_loader, device):
    learning_rate = 1e-2
    num_epochs = 20
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5
    )

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(
            model=model, test_loader=test_loader, device=device, criterion=criterion
        )
        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
            epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))
    return model

def evaluate_model(model, test_loader, device=torch.device("cpu"), criterion=None):
    t0 = time.time()
    model.eval()
    model.to(device)
    running_loss = 0
    running_corrects = 0
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    t1 = time.time()
    print(f"eval loss: {eval_loss}, eval acc: {eval_accuracy}, cost: {t1 - t0}")
    return eval_loss, eval_accuracy
```