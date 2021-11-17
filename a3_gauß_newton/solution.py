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

class Feature(MathematicalProgram):
    def __init__(self,a,c):
        self.a = a
        self.c = c

    def evaluate(self, x):
        f = np.array([np.sin(self.a*x[0]),
                      np.sin(self.a*self.c*x[1]),
                      2*x[0],
                      2*self.c*x[1]])
        y = f.T @ f        
        J = np.array([[self.a*np.cos(self.a*x[0]), 0.0                                     ],
                      [0.0,                        self.a*self.c*np.cos(self.a*self.c*x[1])], 
                      [1.0,                        0.0                                     ],
                      [0.0,                        self.c                                  ]])
        G = 2 * J.T @ f[:,np.newaxis]
        return np.array([y]), G.reshape(1,-1) # f_x, ff_x    

    def getFHessian(self, x):
        J = np.array([[self.a*np.cos(self.a*x[0]), 0.0                                     ],
                      [0.0,                        self.a*self.c*np.cos(self.a*self.c*x[1])], 
                      [1.0,                        0.0                                     ],
                      [0.0,                        self.c                                  ]])
        return 2 * J.T @ J # Gauß-Newton approximation for Hessian

class Solver(NLPSolver):
    def __init__(self, itercnt=100):
        self.tolerance = 1e-4 # theta
        self.max_iter = itercnt 
        self.verbose = False 
        self.testconv = False

    def evaluate(self, x):
        a, b = self.problem.evaluate(x)
        return a[0], b[0]

    def gauss_newton(self, x):
        counter = 0
        iteration = 0
        step_sz = 1.0 # alpha
        step_inc = 1.2 # phi_alpha^plus 
        step_dec = 0.5 # phi_alpha^minus
        step_dec_min = 0.01 # phi_ls 
        eigen_shift = 1.0 # lambda

        if(self.verbose):
            print("commencing gauß-newton method with line search and fallback")

        while self.testconv or iteration < self.max_iter:
            f_x, ff_x = self.evaluate(x) 
            fff_x = self.problem.getFHessian(x)
            shift = eigen_shift * np.eye(x.shape[0])
            
            if(self.verbose):
                print("iteration %d: f(x) = %.5f, alpha = %.5f" % (iteration, f_x, step_sz))
                
            iteration += 1
            step_dir = np.linalg.solve(fff_x + shift, -ff_x)

            try:
                step_dir = np.linalg.solve(fff_x + shift, -ff_x)
                if ff_x @ step_dir > 0.0: step_dir = -ff_x / np.linalg.norm(ff_x) 
            except:
                step_dir = -ff_x / np.linalg.norm(ff_x)             

            if self.testconv and np.linalg.norm(step_sz * step_dir, ord=1) < self.tolerance:
                if counter == 10:
                    break 
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
            step_sz = min(step_inc*step_sz,1)
        return x

    def gradient(self, x):
        counter = 0
        iteration = 0
        step_sz = 1.0 # alpha
        step_inc = 1.2 # phi_alpha^plus 
        step_dec = 0.5 # phi_alpha^minus
        step_dec_min = 0.01 # phi_ls 

        if(self.verbose):
            print("commencing gradient descent with line search")

        while self.testconv or iteration < self.max_iter:
            f_x, ff_x = self.evaluate(x) 

            if(self.verbose):
                print("iteration %d: x = %s, f(x) = %.5f, alpha = %.5f" % (iteration, np.array2string(x,precision=2), f_x, step_sz))
                
            iteration += 1

            step_dir = -ff_x / np.linalg.norm(ff_x)             

            if self.testconv and np.linalg.norm(step_sz * step_dir, ord=1) < self.tolerance:
                if counter == 10:
                    break 
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
        return x
