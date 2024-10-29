import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleMLP(nn.Module):
    """MLP模型结构"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


def train_mlp_model(inferior_solutions, superior_solutions, dim):
    """
    训练MLP模型并保存

    参数:
    inferior_solutions: np.array, 劣汰个体矩阵，形状为(n_inferior, dim)
    superior_solutions: np.array, 优势个体矩阵，形状为(n_superior, dim)
    dim: int, 解向量维度
    """

    # 确保输入数据为numpy数组并转换为tensor
    X_train = torch.FloatTensor(np.array(inferior_solutions))
    y_train = torch.FloatTensor(np.array(superior_solutions))

    # 定义模型参数
    input_size = dim
    hidden_size = dim * 2
    output_size = dim

    # 创建模型实例
    model = SimpleMLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 200
    batch_size = min(32, len(inferior_solutions))

    for epoch in range(epochs):
        # 打乱数据
        indices = torch.randperm(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / (len(X_train) / batch_size)
            print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size
    }, 'mlp_model.pth')

    print("模型已保存到 mlp_model.pth")


def predict_superior_solutions(inferior_solutions):
    """
    加载保存的MLP模型并预测优秀个体

    参数:
    inferior_solutions: np.array, 劣汰个体矩阵，形状为(n_inferior, dim)

    返回:
    predicted_solutions: np.array, 预测的优秀个体，形状与输入相同
    """
    # 加载保存的模型和参数
    checkpoint = torch.load('mlp_model.pth')
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    output_size = checkpoint['output_size']

    # 创建模型实例并加载保存的参数
    model = SimpleMLP(input_size, hidden_size, output_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式

    # 转换输入数据为tensor
    input_tensor = torch.FloatTensor(inferior_solutions)

    # 预测
    with torch.no_grad():
        predicted_tensor = model(input_tensor)
        predicted_solutions = predicted_tensor.numpy()

    return predicted_solutions

def GetCoupleArray(population):
    Rank = 


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    dim = 5
    n_samples = 100

    # 生成随机的劣汰和优势个体
    inferior = np.random.rand(n_samples, dim)
    superior = np.random.rand(n_samples, dim)

    # 训练模型
    train_mlp_model(inferior, superior, dim)

    # 预测新的优秀个体
    test_inferior = np.random.rand(10, dim)
    predicted_superior = predict_superior_solutions(test_inferior)

    print("\n预测结果示例:")
    print("输入的劣汰个体:", test_inferior[0])
    print("预测的优秀个体:", predicted_superior[0])