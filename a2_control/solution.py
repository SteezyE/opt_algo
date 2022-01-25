import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class LQR(MathematicalProgram):
    """
    Parameters
    K integer
    A in R^{n x n}
    B in R^{n x n}
    Q in R^{n x n} symmetric
    R in R^{n x n} symmetric
    yf in R^n

    Variables
    y[k] in R^n for k=1,...,K
    u[k] in R^n for k=0,...,K-1

    Optimization Problem:
    LQR with terminal state constraint

    min 1/2 * sum_{k=1}^{K}   y[k].T Q y[k] + 1/2 * sum_{k=1}^{K-1}      u [k].T R u [k]
    s.t.
    y[1] - Bu[0]  = 0
    y[k+1] - Ay[k] - Bu[k] = 0  ; k = 1,...,K-1
    y[K] - yf = 0

    Hint: Use the optimization variable:
    x = [ u[0], y[1], u[1],y[2] , ... , u[K-1], y[K] ]

    Use the following features:
    1 - a single feature of types OT.f
    2 - the features of types OT.eq that you need
    """

    def __init__(self, K, A, B, Q, R, yf):
        """
        Arguments
        -----
        K: integer
        A: np.array 2-D
        B: np.array 2-D
        Q: np.array 2-D
        R: np.array 2-D
        yf: np.array 1-D
        """
        self.K = K
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Y = yf
        self.N = A.shape[0]
        self.M = K+2
        # in case you want to initialize some class members or so...

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        # y = ...
        # J = ...

        f = 0.0 
        for k in range(1,self.K+1):
            f = f + 0.5 * x[(2*k-2)*self.N:(2*k-1)*self.N].T @ self.R @ x[(2*k-2)*self.N:(2*k-1)*self.N]
            f = f + 0.5 * x[(2*k-1)*self.N:(2*k)*self.N].T @ self.Q @ x[(2*k-1)*self.N:(2*k)*self.N]
        c1 = x[self.N:2*self.N] - self.B @ x[0:self.N]
        ck = np.zeros((self.K-1,self.N))
        for k in range(1,self.K):
            ck[k-1] = x[(2*k+1)*self.N:(2*k+2)*self.N]
            ck[k-1] = ck[k-1] - self.A @ x[(2*k-1)*self.N:(2*k)*self.N]
            ck[k-1] = ck[k-1] - self.B @ x[(2*k)*self.N:(2*k+1)*self.N]
        cK = x[(2*self.K-1)*self.N:(2*self.K)*self.N] - self.Y
        y = np.concatenate((np.array([f]),c1,ck.flatten(),cK))
        ff = np.zeros(self.getDimension())
        for k in range(1,self.K+1):
            ff[(2*k-2)*self.N:(2*k-1)*self.N] = self.R @ x[(2*k-2)*self.N:(2*k-1)*self.N]
            ff[(2*k-1)*self.N:(2*k)*self.N] = self.Q @ x[(2*k-1)*self.N:(2*k)*self.N]
        cc1 = np.zeros((self.N,self.getDimension()))
        cc1[0:self.N,0:self.N] = -1.0 * self.B
        cc1[0:self.N,self.N:2*self.N] = np.eye(self.N) 
        cck = np.zeros((~-self.K * self.N, self.getDimension()))
        for k in range(1,self.K):
            cck[(k-1)*self.N:(k)*self.N,(2*k-1)*self.N:(2*k)*self.N] = -self.A
            cck[(k-1)*self.N:(k)*self.N,(2*k)*self.N:(2*k+1)*self.N] = -self.B
            cck[(k-1)*self.N:(k)*self.N,(2*k+1)*self.N:(2*k+2)*self.N] = np.eye(self.N) 
        ccK = np.zeros((self.N, self.getDimension()))
        ccK[0:self.N,(2*self.K-1)*self.N:(2*self.K)*self.N] = np.eye(self.N)
        J = np.vstack((ff,cc1,cck,ccK))
        # y is a 1-D np.array of dimension m
        # J is a 2-D np.array of dimensions (m,n)
        # where m is the number of features and n is dimension of x
        return y, J

    def getFHessian(self, x):
        """
        """
        # Dimensionality? (2 * self.N * self.K) * (2 * self.N * self.K)
        fff = np.zeros((self.getDimension(),self.getDimension()))
        for k in range(1,self.K+1):
            fff[(2*k-2)*self.N:(2*k-1)*self.N,(2*k-2)*self.N:(2*k-1)*self.N] = self.R
            fff[(2*k-1)*self.N:(2*k)*self.N,(2*k-1)*self.N:(2*k)*self.N] = self.Q
        return fff 

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return 2 * self.N * self.K

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.zeros(self.getDimension())

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.f] + [OT.eq] * (-~self.K * self.N) 
