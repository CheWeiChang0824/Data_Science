import numpy as np
import optuna
from HomeworkFramework import Function

class OptunaOptimizer(Function):
    def __init__(self, target_func):
        super().__init__(target_func)  # 初始化
        self.lower = self.f.lower(target_func)  # 获取下界
        self.upper = self.f.upper(target_func)  # 获取上界
        self.dim = self.f.dimension(target_func)  # 获取维度
        self.target_func = target_func  # 目标函数编号
        self.eval_times = 0  # 评估次数
        self.optimal_value = float("inf")  # 最佳值
        self.optimal_solution = np.empty(self.dim)  # 最佳解
        self.study = optuna.create_study(direction="minimize")  # 创建 Optuna study

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def objective(self, trial):
        solution = np.array([trial.suggest_uniform(f"x{i}", self.lower, self.upper) for i in range(self.dim)])
        value = self.f.evaluate(self.target_func, solution)
        self.eval_times += 1
        if float(value) < self.optimal_value:
            self.optimal_solution[:] = solution
            self.optimal_value = float(value)
        return value

    def run(self, FES):  # 主要实现部分
        self.study.optimize(self.objective, n_trials=FES)

if __name__ == '__main__':
    func_num = 1
    fes = 0
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

        # 你应实现你的优化器
        op = OptunaOptimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(best_input, best_value)

        # 修改文件名为你的学生ID并正确输出
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
