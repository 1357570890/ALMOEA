import numpy as np
from typing import Tuple, List


def fast_non_dominated_sort(fitness: np.ndarray) -> List[np.ndarray]:
    """快速非支配排序"""
    n_pop = len(fitness)
    domination_count = np.zeros(n_pop)
    dominated_solutions = [[] for _ in range(n_pop)]
    fronts = [[]]

    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                dominated_solutions[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front = []
        for j in fronts[i]:
            for k in dominated_solutions[j]:
                domination_count[k] -= 1
                if domination_count[k] == 0:
                    next_front.append(k)
        i += 1
        if next_front:
            fronts.append(next_front)

    return [np.array(front) for front in fronts]


def crowding_distance(fitness: np.ndarray) -> np.ndarray:
    """计算拥挤度距离"""
    n_points, n_objectives = fitness.shape
    distances = np.zeros(n_points)

    for obj in range(n_objectives):
        sorted_idx = np.argsort(fitness[:, obj])
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf

        obj_range = fitness[sorted_idx[-1], obj] - fitness[sorted_idx[0], obj]
        if obj_range == 0:
            continue

        for i in range(1, n_points - 1):
            distances[sorted_idx[i]] += (
                                                fitness[sorted_idx[i + 1], obj] - fitness[sorted_idx[i - 1], obj]
                                        ) / obj_range

    return distances


def nsga2_iteration(population: np.ndarray, fitness: np.ndarray,
                    crossover_rate: float = 0.9, mutation_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    NSGA-II的单次迭代

    参数:
    population: np.ndarray, 当前种群, shape=(pop_size, n_vars)
    fitness: np.ndarray, 适应度值, shape=(pop_size, n_objectives)
    crossover_rate: float, 交叉概率
    mutation_rate: float, 变异概率

    返回:
    new_population: np.ndarray, 新种群
    new_fitness: np.ndarray, 新种群的适应度值
    """
    pop_size = len(population)
    n_vars = population.shape[1]

    # 二进制锦标赛选择
    def binary_tournament(p1, p2, f1, f2):
        fronts1 = fast_non_dominated_sort(np.vstack([f1, f2]))
        if 0 in fronts1[0] and 1 not in fronts1[0]:
            return p1
        elif 1 in fronts1[0] and 0 not in fronts1[0]:
            return p2
        else:
            if np.random.random() < 0.5:
                return p1
            return p2

    # 模拟二进制交叉(SBX)
    def sbx_crossover(p1, p2):
        if np.random.random() > crossover_rate:
            return p1.copy(), p2.copy()

        c1, c2 = p1.copy(), p2.copy()
        eta_c = 20

        for i in range(n_vars):
            if np.random.random() <= 0.5:
                beta = np.random.random()
                if beta <= 0.5:
                    beta = (2 * beta) ** (1 / (eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - beta))) ** (1 / (eta_c + 1))

                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])

                # 边界处理
                c1[i] = min(max(c1[i], 0), 1)
                c2[i] = min(max(c2[i], 0), 1)

        return c1, c2

    # 多项式变异
    def polynomial_mutation(individual):
        eta_m = 20
        child = individual.copy()

        for i in range(n_vars):
            if np.random.random() <= mutation_rate:
                rand = np.random.random()
                if rand < 0.5:
                    delta = (2 * rand) ** (1 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - rand)) ** (1 / (eta_m + 1))

                child[i] += delta
                child[i] = min(max(child[i], 0), 1)

        return child

    # 生成子代
    offspring = np.zeros_like(population)
    for i in range(0, pop_size, 2):
        # 选择父代
        idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
        parent1 = binary_tournament(population[idx1], population[idx2],
                                    fitness[idx1:idx1 + 1], fitness[idx2:idx2 + 1])
        idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
        parent2 = binary_tournament(population[idx1], population[idx2],
                                    fitness[idx1:idx1 + 1], fitness[idx2:idx2 + 1])

        # 交叉和变异
        child1, child2 = sbx_crossover(parent1, parent2)
        child1 = polynomial_mutation(child1)
        child2 = polynomial_mutation(child2)

        if i + 1 < pop_size:
            offspring[i:i + 2] = np.vstack([child1, child2])
        else:
            offspring[i] = child1

    # 合并父代和子代
    combined_pop = np.vstack([population, offspring])
    combined_fitness = np.vstack([fitness, fitness])  # 这里需要计算子代的适应度

    # 非支配排序
    fronts = fast_non_dominated_sort(combined_fitness)

    # 选择下一代种群
    new_population = np.zeros_like(population)
    new_fitness = np.zeros_like(fitness)
    count = 0
    front_idx = 0

    while count + len(fronts[front_idx]) <= pop_size:
        current_front = fronts[front_idx]
        new_population[count:count + len(current_front)] = combined_pop[current_front]
        new_fitness[count:count + len(current_front)] = combined_fitness[current_front]
        count += len(current_front)
        front_idx += 1

    if count < pop_size:
        last_front = fronts[front_idx]
        distances = crowding_distance(combined_fitness[last_front])
        sorted_indices = np.argsort(-distances)
        remaining = pop_size - count
        selected_indices = last_front[sorted_indices[:remaining]]
        new_population[count:] = combined_pop[selected_indices]
        new_fitness[count:] = combined_fitness[selected_indices]

    return new_population, new_fitness


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    pop_size = 100
    n_vars = 10
    n_objectives = 2

    population = np.random.rand(pop_size, n_vars)
    fitness = np.random.rand(pop_size, n_objectives)  # 实际应用中需要计算真实的适应度值

    # 进行一次NSGA-II迭代
    new_pop, new_fitness = nsga2_iteration(population, fitness)