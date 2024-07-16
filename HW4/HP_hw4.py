import numpy as np
from HomeworkFramework import Function

class CMAES_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func) # the lower bound of target function
        self.upper = self.f.upper(target_func) # the upper bound of target function
        self.dim = self.f.dimension(target_func) # the dimension of target function

        self.target_func = target_func # the number of target function

        self.eval_times = 0 # the number of evaluation 
        self.optimal_value = float("inf") 
        self.optimal_solution = np.empty(self.dim) # the best solution in current

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        # Population size
        lam = 2 + int(3 * np.log(self.dim))  # Increased population size
        mu = lam // 2

        # Strategy parameter setting
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = np.sum(weights)**2 / np.sum(weights**2)

        # Adaptation
        sigma = 0.3 * (self.upper - self.lower)
        cs = (mueff + 2) / (self.dim + mueff + 5)
        cc = 4 / (self.dim + 4)
        c1 = 2 / ((self.dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + cs

        # Initialize dynamic (internal) strategy parameters and constants
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        B = np.eye(self.dim)
        D = np.ones(self.dim)
        C = np.eye(self.dim)
        invsqrtC = np.eye(self.dim)
        chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

        mean = np.random.uniform(self.lower, self.upper, self.dim)
        eigeneval = 0

        last_improvement = 0
        patience = 100  # Early stopping patience
        tolerance = 1e-6

        while self.eval_times < FES:
            #print('=====================FE=====================')
            #print(self.eval_times)

            # Generate and evaluate lambda offspring
            arz = np.random.randn(lam, self.dim)
            ary = np.dot(arz, np.diag(D)) @ B.T
            arz = mean + sigma * ary
            arz = np.clip(arz, self.lower, self.upper)

            values = []
            for k in range(lam):
                try:
                    value = self.f.evaluate(self.target_func, arz[k])
                except Exception as e:
                    print(f"Evaluation error: {e}")
                    value = float("inf")

                self.eval_times += 1
                values.append(value)

                if value == "ReachFunctionLimit":
                    print("ReachFunctionLimit")
                    break
                if float(value) < self.optimal_value:
                    self.optimal_solution[:] = arz[k]
                    self.optimal_value = float(value)
                    last_improvement = self.eval_times

            if value == "ReachFunctionLimit":
                break

            # Early stopping
            if self.eval_times - last_improvement > patience:
                print("Early stopping due to lack of improvement.")
                break

            # Sort by fitness and compute weighted mean into mean
            indices = np.argsort(values)
            arz = arz[indices]
            best_ary = ary[indices]
            mean = np.dot(weights, arz[:mu])

            # Cumulation: Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invsqrtC, (mean - self.optimal_solution) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * self.eval_times / lam)) / chiN < 1.4 + 2 / (self.dim + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - self.optimal_solution) / sigma

            # Adapt covariance matrix C
            artmp = (best_ary[:mu].T * np.sqrt(weights)).T  # Correct broadcasting
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * np.dot(artmp.T, artmp)

            # Adapt step size sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

            # Decomposition of C into B*diag(D^2)*B^T (diagonalization)
            if self.eval_times - eigeneval > lam / (c1 + cmu) / self.dim / 10:
                eigeneval = self.eval_times
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eigh(C)
                D = np.sqrt(D)
                invsqrtC = B @ np.diag(1 / D) @ B.T

            #print("optimal: {}\n".format(self.get_optimal()[1]))

if __name__ == '__main__':
    func_num = 1
    fes = 0
    # function1: 1000, function2: 1500, function3: 2000, function4: 2500
    results = []
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
        best_value = float("inf")
        best_input = None
        runs = 5  # Number of runs for averaging
        for run in range(runs):
            op = CMAES_optimizer(func_num)
            op.run(fes)
            current_input, current_value = op.get_optimal()
            if current_value < best_value:
                best_value = current_value
                best_input = current_input

        print(best_input, best_value)
        
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1

