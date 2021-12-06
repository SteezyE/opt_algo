import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class RobotTool(MathematicalProgram):
    """
    """

    def __init__(self, q0, pr, l):
        """
        Arguments
        ----
        q0: 1-D np.array
        pr: 1-D np.array
        l: float
        """
        # in case you want to initialize some class members or so...
        self.q0 = q0
        self.pr = pr
        self.l = l 
        self.lr = math.sqrt(l)
        self.p1 = lambda x: math.cos(x[0])+0.5*math.cos(x[0]+x[1])+math.cos(x[0]+x[1]+x[2])/3.0
        self.p2 = lambda x: math.sin(x[0])+0.5*math.sin(x[0]+x[1])+math.sin(x[0]+x[1]+x[2])/3.0

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        p_x = np.array([self.p1(x), self.p2(x)]) 
        y = np.array([np.linalg.norm(p_x-self.pr), np.linalg.norm(self.lr * (x-self.q0))])
        t_cos = [math.cos(x[0]),0.5*math.cos(x[0]+x[1]),math.cos(x[0]+x[1]+x[2])/3.0][::-1]
        t_sin = [-math.sin(x[0]),-0.5*math.sin(x[0]+x[1]),-math.sin(x[0]+x[1]+x[2])/3.0][::-1]
        J_f = np.array([[sum(t_sin[0:3-i]) for i in range(3)],[sum(t_cos[0:3-i]) for i in range(3)]])
        f_x = p_x-self.pr
        r_x = x-self.q0
        t1 = np.linalg.norm(f_x)
        t2 = np.linalg.norm(self.lr*r_x)
        if t1==0.0 and t2==0.0:
            J = np.zeros((2,self.getDimension()))
        elif t1==0.0:
            J = np.array([np.zeros(self.getDimension()),self.l*r_x/t2])
        elif t2==0.0:
            J = np.array([(J_f.T@f_x)/t1,np.zeros(self.getDimension())])
        else:
            J = np.array([(J_f.T@f_x)/t1,self.l*r_x/t2])
        # J = ...

        # y is a 1-D np.array of dimension m
        # J is a 2-D np.array of dimensions (m,n)
        # where m is the number of features and n is dimension of x
        return y, J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return self.q0.shape[0] 

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return self.q0

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.sos] * 2
