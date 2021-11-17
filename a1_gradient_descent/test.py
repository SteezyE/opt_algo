import numpy as np
import unittest
import sys
sys.path.append("../../..")

from optimization_algorithms.utils.finite_diff import *
from optimization_algorithms.interface.nlp_solver import  NLPSolver
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced

from solution import Gradient, Square, Hole

class testGradientDescent(unittest.TestCase):

    a = 0.1
    C = np.array([[1.0,   0], \
                  [0,  10.0]]) 

    def testSquareFunction(self): 
        print("\n---[testing square function]-----------------------------------------")
        square = MathematicalProgramTraced(Square(self.C)) 
        gradient = Gradient()
        gradient.setProblem((square))
        x_0 = np.array([1.0, 1.0])
        output = gradient.fixed_step_search(x_0)
        self.assertTrue(True)

        x_0 = np.array([1.0, 1.0])
        output = gradient.line_search(x_0)
        self.assertTrue(True)

    def testHoleFunction(self):
        print("\n---[testing hole function]-------------------------------------------")
        hole = MathematicalProgramTraced(Hole(self.C,self.a))
        gradient = Gradient()
        gradient.setProblem((hole))
        x_0 = np.array([1.0, 1.0])
        output = gradient.fixed_step_search(x_0)
        self.assertTrue(True)

        x_0 = np.array([1.0, 1.0])
        output = gradient.line_search(x_0)
        self.assertTrue(True)

if __name__ == "__main__":
   unittest.main()
