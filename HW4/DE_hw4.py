import numpy as np
from HomeworkFramework import Function

class DE_optimizer(Function):  # 繼承 Function 類
    def __init__(self, target_func):
        super().__init__(target_func)  # 必須要有這個 init 才能正常工作

        self.lower = self.f.lower(target_func)  # 目標函數的下界
        self.upper = self.f.upper(target_func)  # 目標函數的上界
        self.dim = self.f.dimension(target_func)  # 目標函數的維度

        self.target_func = target_func  # 目標函數的編號
        # ADD
        print('func_num:', self.target_func)
        print('dim:', self.dim)
        # END
        self.eval_times = 0  # 評估次數
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)  # 當前的最佳解

        # Set population_size dynamically
        if self.dim < 10:
            self.population_size = 25
        elif self.dim < 30:
            self.population_size = 50
        else:
            self.population_size = 100

        self.F = 0.5  # 差分變異的縮放因子
        self.CR = 0.9  # 交叉概率
        self.population = np.random.uniform(self.lower, self.upper, (self.population_size, self.dim))
        self.population_values = np.full(self.population_size, float('inf'))

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def mutate(self, idx):
        idxs = [index for index in range(self.population_size) if index != idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), self.lower, self.upper)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, idx, trial):
        trial_value = self.f.evaluate(self.target_func, trial)
        self.eval_times += 1

        if trial_value == "ReachFunctionLimit":
            print("ReachFunctionLimit")
            return False

        if trial_value < self.population_values[idx]:
            self.population[idx] = trial
            self.population_values[idx] = trial_value

        if trial_value < self.optimal_value:
            self.optimal_solution = trial
            self.optimal_value = trial_value
        
        return True

    def evolution(self):
        for i in range(self.population_size):
            mutant = self.mutate(i)
            trial = self.crossover(self.population[i], mutant)
            if not self.select(i, trial):
                break

    def run(self, FES):  # 差分進化算法的主要實現部分
        while self.eval_times < FES:
            
            # Adjust F and CR dynamically
            if self.eval_times > FES / 2:
                self.F = 0.5
                self.CR = 0.7
            self.evolution()
            print("Evaluation: {}, Current optimal: {}\n".format(self.eval_times, self.optimal_value))
            

if __name__ == '__main__':
    func_num = 1
    fes = 0
    results = []
    # function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500

        # 實現你的優化器
        op = DE_optimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        results.append((best_input, best_value))
        print(f"Function {func_num} best input: {best_input}, best value: {best_value}")

        # 將此文件的名稱更改為你的學生 ID，這樣它會正確輸出
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
    
    # 最後輸出所有函數的最佳結果
    for i, (best_input, best_value) in enumerate(results, 1):
        print(f"Function {i} final best input: {best_input}, final best value: {best_value}")

