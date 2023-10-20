import time

import torch
import torch.nn as nn
from torch import optim

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