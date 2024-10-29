import numpy as np


def guide_individual(inferior: np.ndarray, superior: np.ndarray, learning_rate: float = 0.3) -> np.ndarray:
    """
    引导劣汰个体向优秀个体移动

    参数:
    inferior: np.ndarray, 劣汰个体, shape=(n_vars,)
    superior: np.ndarray, 预测的优秀个体, shape=(n_vars,)
    learning_rate: float, 学习率，控制移动步长，范围(0,1)

    返回:
    guided: np.ndarray, 优化后的个体
    """
    # 计算方向向量
    direction = superior - inferior

    # 引导个体移动
    guided = inferior + learning_rate * direction

    # 确保结果在[0,1]范围内
    guided = np.clip(guided, 0, 1)

    return guided


# 使用示例
if __name__ == "__main__":
    # 示例数据
    n_vars = 10
    inferior_solution = np.random.rand(n_vars)
    superior_solution = np.random.rand(n_vars)

    # 引导优化
    optimized_solution = guide_individual(inferior_solution, superior_solution)

    print("劣汰个体:", inferior_solution)
    print("优秀个体:", superior_solution)
    print("优化后个体:", optimized_solution)