import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys

sys.path.append("../../..")

from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.mathematical_program import MathematicalProgram

class Plot:
    def plotFunc(self, f, bounds_lo, bounds_up, trace_xy = None, trace_z = None):
        x = np.linspace(bounds_lo[0], bounds_up[0], 30)
        y = np.linspace(bounds_lo[1], bounds_up[1], 30)
        xMesh, yMesh = np.meshgrid(x, y, indexing='ij')
        zMesh = np.zeros_like(xMesh)

        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                zMesh[i,j] = f([xMesh[i,j], yMesh[i,j]])

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121, projection="3d")
        surf = ax1.plot_surface(xMesh, yMesh, zMesh, cmap=cm.coolwarm)

        if trace_xy is not None: ax1.plot(trace_xy[:,0], trace_xy[:,1], trace_z[:,0], 'ko-', linewidth=1.0, markersize=2.0)

        fig.colorbar(surf)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f')
        ax2 = fig.add_subplot(122)
        surf2 = plt.contourf(xMesh, yMesh, zMesh, cmap=cm.coolwarm)

        if trace_xy is not None: ax2.plot(trace_xy[:,0], trace_xy[:,1], 'ko-', linewidth=1.0, markersize=2.0)

        fig.colorbar(surf2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        plt.show()

class Square(MathematicalProgram):
    def __init__(self,C):
        self.C = C

    def evaluate(self, x):
        y = x @ self.C @ x       
        J = x @ self.C + x @ self.C.T 
        return np.array([y]), J.reshape(1,-1)
    
    def getFHessian(self, x):
        return 2 * self.C

class Hole(MathematicalProgram):

    def __init__(self, C, a):
        self.C = C
        self.aa = a*a

    def evaluate(self, x):
        y = (x @ self.C @ x) / (x @ self.C @ x + self.aa) 
        J = ((x @ self.C + x @ self.C.T) * self.aa) / (x @ self.C @ x + self.aa)**2
        return np.array([y]), J.reshape(1,-1)

    def getFHessian(self, x):
        t = x @ self.C @ x
        vt = x @ self.C + x @ self.C.T
        return (((2*self.aa*self.C * (t + self.aa)**2) - (vt * self.aa).T * (2 * (t + self.aa) * vt)) / (t + self.aa)**4).T

class Gradient(NLPSolver):
    def __init__(self):
        self.tolerance = 1e-4 # theta
        self.verbose = True

    def evaluate(self, x):
        a, b = self.problem.evaluate(x)
        return a[0], b[0]

    def line_search(self, x, Cinv=None) :
        if Cinv is None: Cinv = np.eye(x.shape[0])
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
                print("iteration %d: f(x) = %.5f, alpha = %.5f" % (iteration, f_x, step_sz))
                iteration += 1

            step_dir = -(Cinv @ ff_x) / np.linalg.norm(ff_x)             

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

    def newton(self, x):
        counter = 0
        iteration = 0
        step_sz = 1.0 # alpha
        step_inc = 1.2 # phi_alpha^plus 
        step_dec = 0.5 # phi_alpha^minus
        step_dec_min = 0.01 # phi_ls 
        eigen_shift = 1.0 # lambda

        if(self.verbose):
            print("commencing newton method with line search and fallback")

        while True:
            f_x, ff_x = self.evaluate(x) 
            fff_x = self.problem.getFHessian(x)
            shift = eigen_shift * np.eye(x.shape[0])
            
            if(self.verbose):
                print("iteration %d: f(x) = %.5f, alpha = %.5f" % (iteration, f_x, step_sz))
                iteration += 1

            try:
                step_dir = np.linalg.solve(fff_x + shift, -ff_x)
                if ff_x @ step_dir[:,np.newaxis] > 0.0: step_dir = -ff_x / np.linalg.norm(ff_x) 
            except:
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
