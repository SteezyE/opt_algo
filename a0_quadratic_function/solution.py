import numpy as np
import sys
sys.path.append("../../..")

from optimization_algorithms.interface.mathematical_program import  MathematicalProgram


class Problem0( MathematicalProgram ):

    def __init__(self,C):
        
        self.C = C
        self.A = self.C.T @ self.C # @ matrix product

    def evaluate(self, x) :

        # compute value y and Jacobian J
        # TODO: x.T transponiert vector nicht vernünftig
        y = x @ self.A @ x[:,np.newaxis] # x[:,np.newaxis] changes dimension (1-D to 2-D) and .T
        J = x @ self.A + x @ self.A.T 
        # tuple of arrays, dim (1) and (1,n)
        # return y, J
        return np.array([y]), J.reshape(1,-1)

    def getDimension(self) : 

        # input dimensionality of the problem
        return 2

    def getFHessian(self, x) : 

        # add code to compute the Hessian matrix
        H = self.A + self.A.T 
        return H

    def getInitializationSample(self) : 

        return np.ones(self.getDimension())

    def report(self , verbose ): 

        return "Quadratic function x C^T C x "
