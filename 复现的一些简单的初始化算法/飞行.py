import numpy as np
from scipy import stats


def levy_flight(current_position: np.ndarray,
                beta: float = 1.5,
                scale: float = 0.01,
                bounds: tuple = (0, 1)) -> np.ndarray:
    """
    莱维飞行

    参数:
    current_position: np.ndarray, 当前位置
    beta: float, 莱维分布的稳定性参数 (1 < beta <= 2)
    scale: float, 步长缩放因子
    bounds: tuple, 变量范围(min, max)

    返回:
    new_position: np.ndarray, 新位置
    """
    size = len(current_position)

    # 生成莱维分布随机数
    u = np.random.normal(0, 1, size)
    v = np.random.normal(0, 1, size)
    step = u / np.power(np.abs(v), 1 / beta)

    # 更新位置
    new_position = current_position + scale * step

    # 边界处理
    new_position = np.clip(new_position, bounds[0], bounds[1])

    return new_position


def random_walk(current_position: np.ndarray,
                step_size: float = 0.1,
                bounds: tuple = (0, 1)) -> np.ndarray:
    """
    随机游走

    参数:
    current_position: np.ndarray, 当前位置
    step_size: float, 步长
    bounds: tuple, 变量范围(min, max)

    返回:
    new_position: np.ndarray, 新位置
    """
    size = len(current_position)

    # 生成随机方向
    direction = np.random.uniform(-1, 1, size)
    direction = direction / np.linalg.norm(direction)

    # 更新位置
    new_position = current_position + step_size * direction

    # 边界处理
    new_position = np.clip(new_position, bounds[0], bounds[1])

    return new_position


def spiral_flight(current_position: np.ndarray,
                  target_position: np.ndarray,
                  a: float = 1.0,
                  b: float = 1.0,
                  r: float = 0.5,
                  bounds: tuple = (0, 1)) -> np.ndarray:
    """
    螺旋飞行

    参数:
    current_position: np.ndarray, 当前位置
    target_position: np.ndarray, 目标位置
    a: float, 螺旋参数a
    b: float, 螺旋参数b
    r: float, 收缩因子
    bounds: tuple, 变量范围(min, max)

    返回:
    new_position: np.ndarray, 新位置
    """
    # 计算距离向量
    distance = target_position - current_position

    # 生成螺旋角度
    theta = np.random.uniform(0, 2 * np.pi)

    # 计算螺旋移动
    spiral_effect = distance * (
            a * np.exp(b * theta) * np.cos(theta) * r
    )

    # 更新位置
    new_position = current_position + spiral_effect

    # 边界处理
    new_position = np.clip(new_position, bounds[0], bounds[1])

    return new_position


def gaussian_random_walk(current_position: np.ndarray,
                         mu: float = 0.0,
                         sigma: float = 0.1,
                         bounds: tuple = (0, 1)) -> np.ndarray:
    """
    高斯随机游走

    参数:
    current_position: np.ndarray, 当前位置
    mu: float, 高斯分布均值
    sigma: float, 高斯分布标准差
    bounds: tuple, 变量范围(min, max)

    返回:
    new_position: np.ndarray, 新位置
    """
    size = len(current_position)

    # 生成高斯随机步长
    step = np.random.normal(mu, sigma, size)

    # 更新位置
    new_position = current_position + step

    # 边界处理
    new_position = np.clip(new_position, bounds[0], bounds[1])

    return new_position


def triangular_walk(current_position: np.ndarray,
                    left: float = -0.1,
                    mode: float = 0.0,
                    right: float = 0.1,
                    bounds: tuple = (0, 1)) -> np.ndarray:
    """
    三角形游走

    参数:
    current_position: np.ndarray, 当前位置
    left: float, 三角分布左端点
    mode: float, 三角分布众数
    right: float, 三角分布右端点
    bounds: tuple, 变量范围(min, max)

    返回:
    new_position: np.ndarray, 新位置
    """
    size = len(current_position)

    # 生成三角分布随机步长
    step = np.random.triangular(left, mode, right, size)

    # 更新位置
    new_position = current_position + step

    # 边界处理
    new_position = np.clip(new_position, bounds[0], bounds[1])

    return new_position


def adaptive_flight(current_position: np.ndarray,
                    generation: int,
                    max_generations: int,
                    flight_type: str = 'levy',
                    target_position: np.ndarray = None,
                    bounds: tuple = (0, 1)) -> np.ndarray:
    """
    自适应飞行策略（根据进化代数自动调整参数）

    参数:
    current_position: np.ndarray, 当前位置
    generation: int, 当前代数
    max_generations: int, 最大代数
    flight_type: str, 飞行类型
    target_position: np.ndarray, 目标位置（用于螺旋飞行）
    bounds: tuple, 变量范围(min, max)

    返回:
    new_position: np.ndarray, 新位置
    """
    # 计算自适应参数
    progress_ratio = generation / max_generations
    adaptive_scale = 0.1 * (1 - progress_ratio)

    if flight_type == 'levy':
        return levy_flight(current_position, scale=adaptive_scale, bounds=bounds)

    elif flight_type == 'random':
        return random_walk(current_position, step_size=adaptive_scale, bounds=bounds)

    elif flight_type == 'spiral' and target_position is not None:
        return spiral_flight(current_position, target_position,
                             r=1 - progress_ratio, bounds=bounds)

    elif flight_type == 'gaussian':
        return gaussian_random_walk(current_position,
                                    sigma=adaptive_scale, bounds=bounds)

    elif flight_type == 'triangular':
        scale = adaptive_scale
        return triangular_walk(current_position,
                               left=-scale, mode=0, right=scale, bounds=bounds)

    else:
        raise ValueError(f"Unknown flight type: {flight_type}")


# 使用示例
if __name__ == "__main__":
    # 生成测试数据
    dim = 10
    current_pos = np.random.rand(dim)
    target_pos = np.random.rand(dim)

    print("当前位置:", current_pos)

    # 测试各种飞行策略
    levy_pos = levy_flight(current_pos)
    print("莱维飞行:", levy_pos)

    random_pos = random_walk(current_pos)
    print("随机游走:", random_pos)

    spiral_pos = spiral_flight(current_pos, target_pos)
    print("螺旋飞行:", spiral_pos)

    gaussian_pos = gaussian_random_walk(current_pos)
    print("高斯随机游走:", gaussian_pos)

    triangular_pos = triangular_walk(current_pos)
    print("三角形游走:", triangular_pos)

    adaptive_pos = adaptive_flight(current_pos, generation=50,
                                   max_generations=100, flight_type='levy')
    print("自适应飞行:", adaptive_pos)