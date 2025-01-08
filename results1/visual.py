import matplotlib.pyplot as plt
import json

# 读取文件内容
with open('trace.txt', 'r') as file:
    lines = file.readlines()

# 解析数据
train_losses = []
train_accs = []
val_losses = []
val_accs = []
epochs = []

for line in lines:
    data = json.loads(line)
    train_losses.append(data['train']['loss'])
    train_accs.append(data['train']['acc'])
    val_losses.append(data['val']['loss'])
    val_accs.append(data['val']['acc'])
    epochs.append(data['epoch'])

# 绘制训练和验证损失
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# 绘制训练和验证准确率
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accs, label='Train Accuracy')
plt.plot(epochs, val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()