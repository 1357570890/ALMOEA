import numpy as np
from scipy import stats


def gaussian_mutation(individual: np.ndarray,
                      sigma: float = 0.1,
                      mutation_rate: float = 0.1,
                      bounds: tuple = (0, 1)) -> np.ndarray:
    """
    高斯变异

    参数:
    individual: np.ndarray, 待变异个体
    sigma: float, 高斯分布的标准差
    mutation_rate: float, 变异概率
    bounds: tuple, 变量范围(min, max)

    返回:
    mutated: np.ndarray, 变异后的个体
    """
    mutated = individual.copy()
    size = len(individual)

    for i in range(size):
        if np.random.random() < mutation_rate:
            # 生成高斯噪声
            noise = np.random.normal(0, sigma)
            mutated[i] += noise

            # 边界处理
            mutated[i] = np.clip(mutated[i], bounds[0], bounds[1])

    return mutated


def t_distribution_mutation(individual: np.ndarray,
                            df: float = 1.0,
                            scale: float = 0.1,
                            mutation_rate: float = 0.1,
                            bounds: tuple = (0, 1)) -> np.ndarray:
    """
    t分布扰动变异

    参数:
    individual: np.ndarray, 待变异个体
    df: float, t分布的自由度
    scale: float, 扰动的缩放因子
    mutation_rate: float, 变异概率
    bounds: tuple, 变量范围(min, max)

    返回:
    mutated: np.ndarray, 变异后的个体
    """
    mutated = individual.copy()
    size = len(individual)

    for i in range(size):
        if np.random.random() < mutation_rate:
            # 生成t分布扰动
            noise = stats.t.rvs(df=df) * scale
            mutated[i] += noise

            # 边界处理
            mutated[i] = np.clip(mutated[i], bounds[0], bounds[1])

    return mutated


def cauchy_mutation(individual: np.ndarray,
                    scale: float = 0.1,
                    mutation_rate: float = 0.1,
                    bounds: tuple = (0, 1)) -> np.ndarray:
    """
    柯西变异

    参数:
    individual: np.ndarray, 待变异个体
    scale: float, 柯西分布的尺度参数
    mutation_rate: float, 变异概率
    bounds: tuple, 变量范围(min, max)

    返回:
    mutated: np.ndarray, 变异后的个体
    """
    mutated = individual.copy()
    size = len(individual)

    for i in range(size):
        if np.random.random() < mutation_rate:
            # 生成柯西分布扰动
            noise = stats.cauchy.rvs(scale=scale)
            mutated[i] += noise

            # 边界处理
            mutated[i] = np.clip(mutated[i], bounds[0], bounds[1])

    return mutated


def differential_mutation(individual: np.ndarray,
                          population: np.ndarray,
                          F: float = 0.5,
                          strategy: str = 'rand/1',
                          bounds: tuple = (0, 1)) -> np.ndarray:
    """
    差分变异

    参数:
    individual: np.ndarray, 待变异个体
    population: np.ndarray, 种群
    F: float, 缩放因子
    strategy: str, 差分策略 ('rand/1', 'best/1', 'rand/2', 'best/2')
    bounds: tuple, 变量范围(min, max)

    返回:
    mutated: np.ndarray, 变异后的个体
    """
    pop_size = len(population)
    mutated = individual.copy()

    if strategy == 'rand/1':
        # 随机选择三个不同的个体
        r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
        mutated = population[r1] + F * (population[r2] - population[r3])

    elif strategy == 'best/1':
        # 使用种群中最好的个体
        best = population[np.random.randint(pop_size)]  # 这里应该根据适应度选择最好的个体
        r1, r2 = np.random.choice(pop_size, 2, replace=False)
        mutated = best + F * (population[r1] - population[r2])

    elif strategy == 'rand/2':
        # 随机选择五个不同的个体
        r1, r2, r3, r4, r5 = np.random.choice(pop_size, 5, replace=False)
        mutated = population[r1] + F * (population[r2] - population[r3]) + F * (population[r4] - population[r5])

    elif strategy == 'best/2':
        # 使用种群中最好的个体
        best = population[np.random.randint(pop_size)]  # 这里应该根据适应度选择最好的个体
        r1, r2, r3, r4 = np.random.choice(pop_size, 4, replace=False)
        mutated = best + F * (population[r1] - population[r2]) + F * (population[r3] - population[r4])

    # 边界处理
    mutated = np.clip(mutated, bounds[0], bounds[1])

    return mutated


def adaptive_mutation(individual: np.ndarray,
                      generation: int,
                      max_generations: int,
                      mutation_type: str = 'gaussian',
                      initial_rate: float = 0.1,
                      final_rate: float = 0.01,
                      bounds: tuple = (0, 1)) -> np.ndarray:
    """
    自适应变异（根据进化代数自动调整变异参数）

    参数:
    individual: np.ndarray, 待变异个体
    generation: int, 当前代数
    max_generations: int, 最大代数
    mutation_type: str, 变异类型 ('gaussian', 't_dist', 'cauchy')
    initial_rate: float, 初始变异率
    final_rate: float, 最终变异率
    bounds: tuple, 变量范围(min, max)

    返回:
    mutated: np.ndarray, 变异后的个体
    """
    # 计算当前的自适应变异率
    current_rate = initial_rate - (initial_rate - final_rate) * (generation / max_generations)

    # 计算当前的自适应扰动强度
    current_scale = 0.1 * (1 - generation / max_generations)

    if mutation_type == 'gaussian':
        return gaussian_mutation(individual, sigma=current_scale,
                                 mutation_rate=current_rate, bounds=bounds)

    elif mutation_type == 't_dist':
        return t_distribution_mutation(individual, scale=current_scale,
                                       mutation_rate=current_rate, bounds=bounds)

    elif mutation_type == 'cauchy':
        return cauchy_mutation(individual, scale=current_scale,
                               mutation_rate=current_rate, bounds=bounds)

    else:
        raise ValueError(f"Unknown mutation type: {mutation_type}")


# 使用示例
if __name__ == "__main__":
    # 生成测试数据
    dim = 10
    individual = np.random.rand(dim)
    population = np.random.rand(50, dim)

    # 测试各种变异
    print("原始个体:", individual)

    gaussian_result = gaussian_mutation(individual)
    print("高斯变异:", gaussian_result)

    t_dist_result = t_distribution_mutation(individual)
    print("t分布变异:", t_dist_result)

    cauchy_result = cauchy_mutation(individual)
    print("柯西变异:", cauchy_result)

    diff_result = differential_mutation(individual, population)
    print("差分变异:", diff_result)

    adaptive_result = adaptive_mutation(individual, generation=50, max_generations=100)
    print("自适应变异:", adaptive_result)