import numpy as np
import nlopt
from HomeworkFramework import Function

class NLoptOptimizer(Function):
    def __init__(self, target_func):
        super().__init__(target_func)  # 初始化
        self.lower = self.f.lower(target_func)  # 获取下界
        self.upper = self.f.upper(target_func)  # 获取上界
        self.dim = self.f.dimension(target_func)  # 获取维度
        self.target_func = target_func  # 目标函数编号
        self.eval_times = 0  # 评估次数
        self.optimal_value = float("inf")  # 最佳值
        self.optimal_solution = np.empty(self.dim)  # 最佳解

        # 选择全局优化算法
        self.global_opt = nlopt.opt(nlopt.GN_CRS2_LM, self.dim)
        self.global_opt.set_lower_bounds([self.lower] * self.dim)
        self.global_opt.set_upper_bounds([self.upper] * self.dim)
        self.global_opt.set_min_objective(self.objective)
        self.global_opt.set_maxeval(500)  # 全局优化最大评估次数

        # 选择局部优化算法
        self.local_opt = nlopt.opt(nlopt.LN_BOBYQA, self.dim)
        self.local_opt.set_lower_bounds([self.lower] * self.dim)
        self.local_opt.set_upper_bounds([self.upper] * self.dim)
        self.local_opt.set_min_objective(self.objective)
        self.local_opt.set_xtol_rel(1e-6)  # 设置相对容差

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def objective(self, x, grad):
        value = self.f.evaluate(self.target_func, x)
        self.eval_times += 1
        if value == "ReachFunctionLimit":
            return float("inf")  # 设置一个极大的值以表示越界
        if float(value) < self.optimal_value:
            self.optimal_solution[:] = x
            self.optimal_value = float(value)
        return float(value)

    def run(self, FES):  # 主要实现部分
        self.global_opt.set_maxeval(FES // 2)  # 分配一半的评估次数给全局优化
        x0 = np.random.uniform(self.lower, self.upper, self.dim)  # 初始猜测
        x_global = self.global_opt.optimize(x0)

        self.local_opt.set_maxeval(FES // 2)  # 分配另一半的评估次数给局部优化
        self.local_opt.optimize(x_global)

if __name__ == '__main__':
    func_num = 1
    fes = 0
    num_runs = 10  # 设置运行次数
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

        best_values = []
        best_parameters = []

        for run_id in range(num_runs):
            # 你应实现你的优化器
            op = NLoptOptimizer(func_num)
            op.run(fes)

            best_input, best_value = op.get_optimal()
            best_values.append(best_value)
            best_parameters.append(best_input)
            print(f"Run {run_id+1}: Current Value = {best_value}, Parameters = {best_input}")

        avg_value = np.mean(best_values)
        min_value = np.min(best_values)
        min_index = np.argmin(best_values)
        best_params_overall = best_parameters[min_index]

        print(f"\nFunction {func_num} Results:")
        print(f"Average Best Value: {avg_value}")
        print(f"Minimum Best Value: {min_value}\n")

        results.append((avg_value, min_value))

        # 修改文件名为你的学生ID并正确输出
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(len(best_params_overall)):
                f.write("{}\n".format(best_params_overall[i]))
            f.write("{}\n".format(min_value))
        
        func_num += 1

    # 打印所有函数的结果
    for i, result in enumerate(results):
        print(f"Function {i+1}: Average Best Value = {result[0]}, Minimum Best Value = {result[1]}")
