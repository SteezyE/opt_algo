import numpy as np
import sys
sys.path.append("../../..")

from optimization_algorithms.interface.nlp_solver import  NLPSolver

class Solver0(NLPSolver):
    def evaluate(self, x):
        a, b = self.problem.evaluate(x)
        return a[0], b[0]

    def solve(self) :
        x = self.problem.getInitializationSample()

        step_sz = 1.0 # alpha
        step_inc = 1.2 # phi_alpha^plus 
        step_dec = 0.5 # phi_alpha^minus
        step_dec_min = 0.01 # phi_ls 
        tolerance = 1e-6 # theta

        debug = False

        while True:
            f_x, ff_x = self.evaluate(x) 
            step_dir = - ff_x / np.linalg.norm(ff_x) # L2-Norm: ord=2 
            f_xs, ff_xs = self.evaluate(x + step_sz * step_dir)
            while np.all(np.greater(f_xs, f_x + step_dec_min * np.transpose(ff_x) @ (step_sz * step_dir))):
                f_x, ff_x = self.evaluate(x) 
                f_xs, ff_xs = self.evaluate(x + step_sz * step_dir)
                step_sz = step_sz * step_dec 
            x = x + step_sz * step_dir
            step_sz = step_sz * step_inc
            if debug:
                print(np.linalg.norm(step_sz * step_dir))
            if np.linalg.norm(step_sz * step_dir, ord=1) < tolerance:
                if debug:
                    print(x)
                return x

        # use the following to get an initialization:
        # x = self.problem.getInitializationSample()

        # use the following to query the problem:
        # phi, J = self.problem.evaluate(x)

        # phi is a vector (1D np.array). 
        # use phi[0] to access the cost value (a float number). 

        # J is a Jacobian matrix (2D np.array). 
        # Use J[0] to access the gradient (1D np.array) of the cost value.
