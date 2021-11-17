import numpy as np
import unittest
import sys
sys.path.append("../../..")

from optimization_algorithms.utils.finite_diff import *
from optimization_algorithms.interface.nlp_solver import  NLPSolver
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced

from solution import Solver, Feature, Plot 

class testGradientDescent(unittest.TestCase):

    itercnt = 1000
    dim = 2 
    A = [4.0, 5.0] 
    C = [3.0]

    def testGaussNewton(self): 
        for a in self.A:
            for c in self.C:
                print("\n---[testing Gauß-Newton (a=%d, c=%d)]---" % (a,c))
                feature = MathematicalProgramTraced(Feature(a,c)) 
                feat = Feature(a,c)
                solver = Solver(self.itercnt)
                solver.setProblem((feature))
                x_0 = np.array([np.random.uniform(-1.0, 1.0) for x in range(self.dim)])
                h = lambda x: feat.evaluate(x)[0][0]
                print(f"x_0 = {x_0}, f(x_0) = {h(x_0)}")
                solver.gauss_newton(x_0)
                plotter = Plot()
                trace_x = np.array(feature.trace_x)
                trace_phi = np.array(feature.trace_phi)                
                #print(f"x_{self.itercnt/2} = {trace_x[self.itercnt/2]}, f(x_0) = {trace_phi[self.itercnt/2]}")
                print(f"x_{self.itercnt} = {trace_x[-1]}, f(x_{self.itercnt}) = {trace_phi[-1][0]}")
                plotter.plotFunc(h, [-2,-2], [2,2], trace_x, trace_phi)
                self.assertTrue(True)

    def testGradientDescent(self): 
        for a in self.A:
            for c in self.C:
                print("\n---[testing Gradient descent (a=%d, c=%d)]---" % (a,c))
                feature = MathematicalProgramTraced(Feature(a,c)) 
                feat = Feature(a,c)
                solver = Solver(self.itercnt)
                solver.setProblem((feature))
                x_0 = np.array([np.random.uniform(-1.0, 1.0) for x in range(self.dim)])
                h = lambda x: feat.evaluate(x)[0][0]
                print(f"x_0 = {x_0}, f(x_0) = {h(x_0)}")
                solver.gradient(x_0)
                plotter = Plot()
                trace_x = np.array(feature.trace_x)
                trace_phi = np.array(feature.trace_phi)
                #print(f"x_{self.itercnt/2} = {trace_x[self.itercnt/2]}, f(x_0) = {trace_phi[self.itercnt/2]}")
                print(f"x_{self.itercnt} = {trace_x[-1]}, f(x_{self.itercnt}) = {trace_phi[-1][0]}")
                plotter.plotFunc(h, [-2,-2], [2,2], trace_x, trace_phi)
                self.assertTrue(True)

if __name__ == "__main__":
   unittest.main()
