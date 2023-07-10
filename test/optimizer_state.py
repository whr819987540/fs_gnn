import torch
from torch import nn

# 定义模型和优化器
model = nn.Linear(2, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def process_gradient(grad):
    return grad

# 定义钩子函数
def gradient_hook(grad):
    # 周期性地处理梯度
    # 例如，每隔5个训练周期处理一次梯度
    print(optimizer.state['step'])
    print(optimizer.state_dict())

    if optimizer.state['step'] % 5 == 0:
        # 处理梯度
        processed_grad = process_gradient(grad)
        return processed_grad
    else:
        return grad

# 为模型参数注册钩子函数
for param in model.parameters():
    param.register_hook(gradient_hook)

optimizer.state['step'] = 0
print(optimizer.state_dict())

# 训练循环
x = torch.Tensor([[1,3],[2,4]])
y = torch.Tensor([0,1])
y = y.long()
print(x.dtype,y)
loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
for epoch in range(10):
    optimizer.zero_grad()
    # 前向传播和计算损失
    loss = loss_func(model(x), y)
    # 反向传播
    loss.backward()
    optimizer.step()

    optimizer.state['step'] += 1