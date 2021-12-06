import math
import sys
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class AntennaPlacement(MathematicalProgram):
    """
    """

    def __init__(self, P, w):
        """
        Arguments
        ----
        P: list of 1-D np.arrays
        w: 1-D np.array
        """
        # in case you want to initialize some class members or so...
        self.p = P 
        self.dim = P[0].shape[0]
        self.w = w
        self.m = w.shape[0]

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        y = np.array([-sum([self.w[i] * math.exp(-((x-self.p[i]).T@(x-self.p[i]))) for i in range(self.m)])])
        J = np.zeros(self.dim) 
        for i in range(self.m):
            J = np.add(J,2.0*self.w[i]*math.exp((-x+self.p[i]).T@(x-self.p[i]))*(x-self.p[i]))
        return y, J.reshape(1,-1)
        # TODO: Hessian


    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        # return ...
        return self.dim

    def getFHessian(self, x):
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        # add code to compute the Hessian matrix
        y = 0.0
        for i in range(self.m):
            y += self.w[i] * math.exp(-((x-self.p[i]).T@(x-self.p[i])))
        H = 2.0*y*np.eye(self.dim)
        for i in range(self.m):
            H = H-(4.0*y*np.outer(x-self.p[i],x-self.p[i])) 
        return H


    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        # x0 = ...
        # return x0
        return 1.0/self.m * np.sum(self.p)

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types

        """
        return [OT.f]
