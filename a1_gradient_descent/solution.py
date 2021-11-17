import numpy as np
import sys
sys.path.append("../../..")

from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.mathematical_program import MathematicalProgram

class Square(MathematicalProgram):

    def __init__(self,C):
        self.C = C

    def evaluate(self, x) :
        y = x @ self.C @ x[:,np.newaxis]         
        J = x @ self.C + x @ self.C.T 
        return np.array([y]), J.reshape(1,-1)

class Hole(MathematicalProgram):

    def __init__(self, C, a):
        self.C = C
        self.aa = a*a

    def evaluate(self, x) :
        y = (x @ self.C @ x[:,np.newaxis]) / (x @ self.C @ x[:,np.newaxis] + self.aa) 
        J = ((x @ self.C + x @ self.C.T) * ((x @ self.C @ x[:,np.newaxis] + self.aa) - (x @ self.C @ x[:,np.newaxis]))) / (x @ self.C @ x[:,np.newaxis] + self.aa)**2
        return np.array([y]), J.reshape(1,-1)

class Gradient(NLPSolver):
    def __init__(self):
        self.tolerance = 1e-4 # theta
        self.verbose = True

    def evaluate(self, x):
        a, b = self.problem.evaluate(x)
        return a[0], b[0]

    def fixed_step_search(self, x):
        counter = 0
        iteration = 0
        step_sz = np.random.uniform(0.01, 0.02) # alpha

        if(self.verbose): 
            print("commencing gradient descent with fixed stepsize (alpha = %.5f)" % step_sz)

        while True:
            f_x, ff_x = self.evaluate(x)

            if(self.verbose):
                print("iteration %d: x = %s, f(x) = %.5f" % (iteration, np.array2string(x,precision=2), f_x))
            
            iteration += 1
            
            if np.linalg.norm(step_sz * ff_x, ord=1) < self.tolerance:
                if counter == 10:
                    return x
                else:
                    counter += 1
            else: 
                counter = 0

            x -= step_sz * ff_x 

    def line_search(self, x) :
        counter = 0
        iteration = 0
        step_sz = 1.0 # alpha
        step_inc = 1.2 # phi_alpha^plus 
        step_dec = 0.5 # phi_alpha^minus
        step_dec_min = 0.01 # phi_ls 

        if(self.verbose):
            print("commencing gradient descent with line search")

        while True:
            f_x, ff_x = self.evaluate(x) 

            if(self.verbose):
                print("iteration %d: x = %s, f(x) = %.5f, alpha = %.5f" % (iteration, np.array2string(x,precision=2), f_x, step_sz))
                iteration += 1

            step_dir = -ff_x / np.linalg.norm(ff_x)             

            if np.linalg.norm(step_sz * step_dir, ord=1) < self.tolerance:
                if counter == 10:
                    return x
                else:
                    counter += 1
            else: 
                counter = 0

            f_xs, ff_xs = self.evaluate(x + step_sz * step_dir)
            while np.all(np.greater(f_xs, f_x + step_dec_min * np.transpose(ff_x) @ (step_sz * step_dir))):
                f_x, ff_x = self.evaluate(x) 
                f_xs, ff_xs = self.evaluate(x + step_sz * step_dir)
                step_sz *= step_dec 
            x += step_sz * step_dir
            step_sz *= step_inc
