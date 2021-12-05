import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverUnconstrained(NLPSolver):

    def __init__(self):
        """ See also:
        ----
        NLPSolver.__init__
        """
        # in case you want to initialize some class members or so...
        self.ot = np.array([]) 

    def evaluate(self, x):
        a, b = self.problem.evaluate(x)
        return a[0], b[0]

    def evalfsos(self, x):
        phi, J = self.problem.evaluate(x)
        g_x = sum([phi[i][0] if x == OT.f else phi[i].T@phi[i] for i, x in enumerate(self.ot)]) 
        index_r = [i for i, x in enumerate(self.ot) if x == OT.sos] 
        index_f = [i for i, x in enumerate(self.ot) if x == OT.f]
        pp_x = 2.0 * J[index_r].T @ phi[i]
        ff_x = J[index_f]
        gg_x = ff_x + pp_x
        return g_x, gg_x

    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        # write your code here
        # use the following to get an initialization:
        x = self.problem.getInitializationSample()
        n = x.shape[0]
        # get feature types
        self.ot = self.problem.getFeatureTypes()
        # ot[i] inidicates the type of feature i (either OT.f or OT.sos)
        # there is at most one feature of type OT.f
        

        # use the following to query the problem:
        # phi, J = self.problem.evaluate(x)
        # H = self.problem.getFHessian(x)  # if necessary
        # phi is a vector (1D np.array); J is a Jacobian matrix (2D np.array).

        # now code some loop that iteratively queries the problem and updates x til convergenc....
        if OT.f in self.ot and OT.sos in self.ot: 
            # Problem Type: x* = argmin f(x) + phi(x)^T * phi(x)
            # (BFGS, Gradient Descent, other)? 
            # Gradient Descent?!
            __ = 0
            it = 0
            alpha = 1.0
            phi_ls = 0.01 
            alpha_inc = 1.2
            alpha_dec = 0.5
            for _ in range(1000):
                g_x, gg_x = self.evalfsos(x) 
                delta = -gg_x / np.linalg.norm(gg_x) 
                g_xn, gg_xn = self.evalfsos(x + alpha * delta)
                while np.all(np.greater(g_xn, g_x + phi_ls * gg_x.T @ (alpha * delta))):
                    alpha *= alpha_dec
                    g_xn, gg_xn = self.evalfsos(x + alpha * delta)
                    __ += 1
                if np.linalg.norm(alpha * delta) <= 0.001: 
                    it += 1
                    if it == 5:
                        print("Function Calls: %d"%(_+__))
                        return x
                else:
                    it = 0
                x = x + alpha * delta 
                alpha = min(alpha_inc * alpha, 1.0)
        elif OT.sos in self.ot:
            # Problem Type: x* = argmin phi(x)^T * phi(x)
            # Gauß-Newton Approximation
            __ = 0
            it = 0
            alpha = 1.0
            phi_ls = 0.01 
            alpha_inc = 1.2
            alpha_dec = 0.5
            for _ in range(1000):
                phi, J = self.problem.evaluate(x)
                f_x, ff_x, fff_x = phi.T * phi, 2.0*J.T@phi, 2.0*J.T@J  
                lamb = max(0.0,-np.amin(np.linalg.eig(fff_x)[0])+0.001)
                damp = lamb * np.eye(n)
                try:
                    delta = np.linalg.solve(fff_x + damp,-ff_x)
                    if ff_x.T @ delta > 0.0: delta = -ff_x / np.linalg.norm(ff_x) 
                except:
                    delta = -ff_x / np.linalg.norm(ff_x)

                phi, J = self.problem.evaluate(x + alpha * delta)
                while np.all(np.greater(phi.T * phi, f_x + phi_ls * ff_x.T @ (alpha * delta))):
                    alpha *= alpha_dec
                    phi, J = self.problem.evaluate(x + alpha * delta)
                    __ += 1
                if np.linalg.norm(alpha * delta) <= 0.001: 
                    it += 1
                    if it == 5:
                        print("Function Calls: %d"%(_+__))
                        return x
                else:
                    it = 0
                x = x + alpha * delta 
                alpha = min(alpha_inc * alpha, 1.0)
        else:
            # Problem Type: x* = argmin f(x)
            # if f(x) is convex <=> H(x) is positive definite (np.linalg.eigen)
            #   Newton Method: 2 steps should be enough 
            # else: (Newton, BFGS, Gradient Descent, other)?
            #   Newton with damping?!
            __ = 0
            it = 0
            alpha = 1.0
            phi_ls = 0.01 
            alpha_inc = 1.2
            alpha_dec = 0.5
            for _ in range(1000):
                f_x, ff_x = self.evaluate(x)
                fff_x = self.problem.getFHessian(x)
                lamb = max(0.0,-np.amin(np.linalg.eig(fff_x)[0])+0.001)
                damp = lamb * np.eye(fff_x.shape[0])
                try:
                    delta = np.linalg.solve(fff_x + damp,-ff_x)
                    if ff_x.T @ delta > 0.0: delta = -ff_x / np.linalg.norm(ff_x) 
                except:
                    delta = -ff_x / np.linalg.norm(ff_x)
                f_xn, ff_xn = self.evaluate(x + alpha * delta)
                while np.all(np.greater(f_xn, f_x + phi_ls * ff_x.T @ (alpha * delta))):
                    alpha *= alpha_dec
                    f_xn, ff_xn = self.problem.evaluate(x + alpha * delta.T)
                    __ += 1
                if np.linalg.norm(alpha * delta) <= 0.001: 
                    it += 1
                    if it == 5:
                        print("Function Calls: %d"%(_+__))
                        return x
                else:
                    it = 0
                x = x + alpha * delta 
                alpha = min(alpha_inc * alpha, 1.0)
        return x
