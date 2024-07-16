import numpy as np
from HomeworkFramework import Function

class ADAM_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func) # the upper bound of target function
        self.upper = self.f.upper(target_func) # the lower bound of target function
        self.dim = self.f.dimension(target_func) # the dimension of target function

        self.target_func = target_func # the number of target function
        # ADD
        print('here is some info about function')
        print('num:', self.target_func)
        print('dim:', self.dim)
        print('upper bound:', self.upper)
        print('lower bound:', self.lower)
        # END
        self.eval_times = 0 # the number of evaluation 
        self.optimal_value = float("inf") 
        self.optimal_solution = np.empty(self.dim) # the best solution in current

        self.m = np.zeros(self.dim)  # 1st moment vector
        self.v = np.zeros(self.dim)  # 2nd moment vector
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.learning_rate = 0.01

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def gradient(self, x):
        grad = np.zeros(self.dim)
        h = 1e-8  # A small number for numerical gradient calculation
        for i in range(self.dim):
            x1 = np.array(x)
            x2 = np.array(x)
            x1[i] += h
            x2[i] -= h
            val1 = self.f.evaluate(self.target_func, x1)
            val2 = self.f.evaluate(self.target_func, x2)
            if isinstance(val1, (float, int)) and isinstance(val2, (float, int)):
                grad[i] = (val1 - val2) / (2 * h)
            else:
                grad[i] = 0
        return grad

    def run(self, FES): # main part for your implementation
        solution = np.random.uniform(self.lower, self.upper, self.dim)
        t = 0
        
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)

            grad = self.gradient(solution)
            t += 1
            
            # Update biased first moment estimate
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = self.m / (1 - self.beta1 ** t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v / (1 - self.beta2 ** t)

            # Update solution
            solution -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            solution = np.clip(solution, self.lower, self.upper)
            
            value = self.f.evaluate(self.target_func, solution)
            self.eval_times += 1

            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break            
            if isinstance(value, (float, int)) and float(value) < self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)

            print("optimal: {}\n".format(self.get_optimal()[1]))

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = ADAM_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
