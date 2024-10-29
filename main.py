import numpy as np
import random


def calFitness(PopObj):
    """计算适应度"""
    PopObj = np.array(PopObj)
    N = PopObj.shape[0]
    fmax = np.max(PopObj, axis=0)
    fmin = np.min(PopObj, axis=0)
    PopObj = (PopObj - fmin) / (fmax - fmin)
    Dis = np.full((N, N), np.inf)

    for i in range(N):
        SPopObj = np.maximum(PopObj, PopObj[i])
        for j in range(N):
            if j != i:
                Dis[i, j] = np.linalg.norm(PopObj[i] - SPopObj[j])

    Fitness = np.min(Dis, axis=1)
    return Fitness


'''
以下函数需要自行实现:
1. UniformPoint(N, M)
2. Problem.Initialization()
3. isTerminated(Population)
4. classification(Population, ideal, nadir)
5. BPNN(decs, labelset_decs, population_decs)
6. OperatorDE1(ind, p1, p2, lx, t)
7. OperatorDE(ind, p1, p2)
8. NDSort(objs, cons, n)
9. EnvironmentalSelection1(population, n)
10. Operator(loser, winner)
11. EnvironmentalSelection(population, v, t)
"""

if __name__ == "__main__":

    N = 100  # 种群大小
    M = 2  # 目标数量
    FE = 0  # 当前评价次数
    maxFE = 1000  # 最大评价次数

    V,N = UniformPoint(N, M)  #根据种群大小和目标数量生成初始解
    Population = Initialization()

    # 初始化参数
    ideal = 1e17
    nadir = -1e18
    eNum1 = np.zeros(100)
    count = 0
    flag = False
    z, znad = np.min([p.objs for p in Population], axis=0), np.max([p.objs for p in Population], axis=0)

    # 优化过程
    while not isTerminated(Population):
        while not flag:
            eNum = 0
            # 分类
            P, LabelSet, ideal, nadir = classification(Population, ideal, nadir)
            t = Problem.FE / Problem.maxFE

            # 训练
            Lx = BPNN(P.decs, np.tile(LabelSet.decs, (P.decs.shape[0], 1)), Population.decs)

            # 生成子代
            Offspring1 = []
            for i in range(Problem.N):
                if random.random() < 0.5:
                    P1, P2 = random.sample(range(Problem.N), 2)
                    offspring = OperatorDE1(Population[i], Population[P1], Population[P2], Lx[i], t)
                else:
                    P1, P2 = random.sample(range(Problem.N), 2)
                    offspring = OperatorDE(Population[i], Population[P1], Population[P2])
                Offspring1.append(offspring)

            Offspring = Offspring1

            # 评估子代
            for i in range(Problem.N):
                Population1 = [Offspring[i], Population[i]]
                FrontNo, MaxFNo = NDSort([ind.objs for ind in Population1],
                                         [ind.cons for ind in Population1],
                                         Problem.N + 1)
                if FrontNo[-1] != 1:
                    eNum += 1

            eNum1[count] = eNum / Problem.N

            if eNum1[count] < 0.15:
                flag = True

            Population = EnvironmentalSelection1(Population + Offspring, Problem.N)
            count += 1

            if Problem.FE >= 1e6:
                flag = True

        # 确保种群大小
        while len(Population) < 4:
            a = len(Population)
            Population.append(random.choice(Population))

        # 计算适应度
        Fitness = calFitness([ind.objs for ind in Population])

        # 竞争选择
        if len(Population) >= 2:
            Rank = random.sample(range(len(Population)), len(Population))
            Loser = Rank[:len(Rank) // 2]
            Winner = Rank[len(Rank) // 2:]
        else:
            Loser, Winner = [0], [0]

        Change = Fitness[Loser] >= Fitness[Winner]
        Temp = np.array(Winner)[Change].tolist()
        Winner = np.array(Winner)
        Loser = np.array(Loser)
        Winner[Change] = Loser[Change]
        Loser[Change] = Temp

        # 生成新的后代
        Offspring = Operator([Population[i] for i in Loser],
                             [Population[i] for i in Winner])

        # 环境选择
        Population = EnvironmentalSelection(Population + Offspring ,V ,(Problem.FE / Problem.maxFE) ** 2)

