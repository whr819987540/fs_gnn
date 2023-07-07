import torch
from sklearn.metrics import accuracy_score

# 假设有真实标签和预测概率分布
true_labels = torch.tensor([[0, 1, 1],
                            [1, 0, 1],
                            [0, 0, 1],
                            [1, 1, 0]])
predicted_probs = torch.tensor([[0.8, 0.9, 0.3],
                               [0.2, 0.7, 0.5],
                               [0.4, 0.3, 0.6],
                               [0.6, 0.5, 0.2]])

# 对预测概率进行二值化，根据阈值确定最终预测
threshold = 0.5
predicted_labels = (predicted_probs >= threshold).int()
print(predicted_labels)

# 计算准确率
accuracy = accuracy_score(true_labels.view(-1), predicted_labels.view(-1))

# 打印准确率
print("Accuracy:", accuracy)
