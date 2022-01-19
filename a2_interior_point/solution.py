import numpy as np
import math  as ma
import copy  as cp
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverInteriorPoint(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        self.ec = 0
        self.ot = np.array([])

    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        self.ec = 0
        self.ot = self.problem.getFeatureTypes()
        x = self.problem.getInitializationSample()
        n = x.shape[0]
        if OT.f not in self.ot and OT.sos in self.ot and OT.ineq not in self.ot:
            # x* = argmin phi(x)^T @ phi(x)
            # Quasi-Newton
            __ = 0
            it = 0
            alpha = 1.0
            phi_ls = 0.01 
            alpha_inc = 1.2
            alpha_dec = 0.5
            for _ in range(1000):
                phi, J = self.problem.evaluate(x)
                if self.ec >= 9990:
                   return x 
                else:
                    self.ec += 1
                f_x, ff_x, fff_x = phi.T * phi, 2.0*J.T@phi, 2.0*J.T@J  
                lamb = max(0.0,-np.amin(np.linalg.eig(fff_x)[0])+0.001)
                damp = lamb * np.eye(n)
                try:
                    delta = np.linalg.solve(fff_x + damp,-ff_x)
                    if ff_x.T @ delta > 0.0: delta = -ff_x / np.linalg.norm(ff_x) 
                except:
                    delta = -ff_x / np.linalg.norm(ff_x)
                phi, J = self.problem.evaluate(x + alpha * delta)
                if self.ec >= 9990:
                    return x 
                else:
                    self.ec += 1
                while np.all(np.greater(phi.T * phi, f_x + phi_ls * ff_x.T @ (alpha * delta))):
                    alpha *= alpha_dec
                    phi, J = self.problem.evaluate(x + alpha * delta)
                    if self.ec >= 9990:
                        return x 
                    else:
                        self.ec += 1
                    __ += 1
                if np.linalg.norm(alpha * delta) <= 0.0001: 
                    it += 1
                    if it == 5:
                        break 
                else:
                    it = 0
                x = x + alpha * delta 
                alpha = min(alpha_inc * alpha, 1.0)
        if OT.f not in self.ot and OT.sos in self.ot and OT.ineq in self.ot:
            # x* = argmin phi(x)^T @ phi(x) s.t. g(x) <= 0
            # Log Barrier w/ Quasi-Newton
            mu = 1.0
            l = 400 
            _it = 0
            def eval_help(x):
                succ = True
                phi, J = self.problem.evaluate(x)
                c_x, cc_x, ccc_x = np.array([]), np.array([]), np.array([])
                index_s = [i for i, x in enumerate(self.ot) if x == OT.sos] 
                index_g = [i for i, x in enumerate(self.ot) if x == OT.ineq]
                try:
                    c_x = phi[index_s].T @ phi[index_s] - mu * sum([ma.log(-phi[i]) for i in index_g]) 
                    cc_x = 2.0 * J[index_s].T @ phi[index_s] - mu * sum([J[i]/phi[i] for i in index_g])
                    ccc_x = 2.0 * J[index_s].T @ J[index_s]
                except:
                    succ = False
                return succ, c_x, cc_x, ccc_x
            for ___ in range(10000//l):
                x_old = cp.deepcopy(x)
                __ = 0
                it = 0
                alpha = 1.0
                phi_ls = 0.01 
                alpha_inc = 1.2
                alpha_dec = 0.5
                for _ in range(l):
                    flag, f_x, ff_x, fff_x = eval_help(x)  
                    if self.ec >= 9990:
                        return x 
                    else:
                        self.ec += 1
                    lamb = max(0.0,-np.amin(np.linalg.eig(fff_x)[0])+0.001)
                    damp = lamb * np.eye(n)
                    try:
                        delta = np.linalg.solve(fff_x + damp,-ff_x)
                        if ff_x.T @ delta > 0.0: delta = -ff_x / np.linalg.norm(ff_x) 
                    except:
                        delta = -ff_x / np.linalg.norm(ff_x)
                    flag_n, f_xn, ff_xn, fff_xn = eval_help(x + alpha * delta)
                    if self.ec >= 9990:
                        return x 
                    else:
                        self.ec += 1
                    while not flag_n or np.all(np.greater(f_xn, f_x + phi_ls * ff_x.T @ (alpha * delta))):
                        alpha *= alpha_dec
                        flag_n, f_xn, ff_xn, fff_xn = eval_help(x + alpha * delta)
                        if self.ec >= 9990:
                            return x 
                        else:
                            self.ec += 1
                        if __ < l: 
                            __ += 1
                        else: 
                            break
                    if np.linalg.norm(alpha * delta) <= 0.0001: 
                        it += 1
                        if it == 5:
                           break 
                    else:
                        it = 0
                    x = x + alpha * delta 
                    alpha = min(alpha_inc * alpha, 1.0)    
                mu = 0.5 * mu
                if np.linalg.norm(x-x_old) <= 0.0001: 
                    _it += 1
                    if _it == 5:
                        break 
                else:
                    _it = 0
        if OT.f in self.ot and OT.sos not in self.ot and OT.ineq not in self.ot:
            # x* = argmin f(x)
            # Newton
            def eval_help(x):
                a, b = self.problem.evaluate(x) 
                return a[0], b[0]
            __ = 0
            it = 0
            alpha = 1.0
            phi_ls = 0.01 
            alpha_inc = 1.2
            alpha_dec = 0.5
            for _ in range(1000):
                f_x, ff_x = eval_help(x)
                if self.ec >= 9990:
                    return x
                else:
                    self.ec += 1
                fff_x = self.problem.getFHessian(x)
                if self.ec >= 9990:
                    return x
                else:
                    self.ec += 1
                lamb = max(0.0,-np.amin(np.linalg.eig(fff_x)[0])+0.001)
                damp = lamb * np.eye(fff_x.shape[0])
                try:
                    delta = np.linalg.solve(fff_x + damp,-ff_x)
                    if ff_x.T @ delta > 0.0: delta = -ff_x / np.linalg.norm(ff_x) 
                except:
                    delta = -ff_x / np.linalg.norm(ff_x)
                f_xn, ff_xn = eval_help(x + alpha * delta)
                if self.ec >= 9990:
                    return x
                else:
                    self.ec += 1
                while np.all(np.greater(f_xn, f_x + phi_ls * ff_x.T @ (alpha * delta))):
                    alpha *= alpha_dec
                    f_xn, ff_xn = eval_help(x + alpha * delta.T)
                    if self.ec >= 9990:
                        return x
                    else:
                        self.ec += 1
                    __ += 1
                if np.linalg.norm(alpha * delta) <= 0.0001: 
                    it += 1
                    if it == 5:
                       break 
                else:
                    it = 0
                x = x + alpha * delta 
                alpha = min(alpha_inc * alpha, 1.0)
        if OT.f in self.ot and OT.sos not in self.ot and OT.ineq in self.ot:
            # x* = argmin f(x) s.t. g(x) <= 0
            # Log Barrier w/ Quasi-Newton
            mu = 1.0
            l = 400 
            _it = 0
            def eval_help(x):
                succ = True
                phi, J = self.problem.evaluate(x)
                c_x, cc_x, ccc_x = np.array([]), np.array([]), np.array([])
                index_f = [i for i, x in enumerate(self.ot) if x == OT.f]
                index_g = [i for i, x in enumerate(self.ot) if x == OT.ineq]
                try:
                    c_x = sum([phi[i] for i in index_f]) - mu * sum([ma.log(-phi[i]) for i in index_g]) 
                    cc_x = J[index_f][0] - mu * sum([J[i]/phi[i] for i in index_g])
                    ccc_x = self.problem.getFHessian(x)
                except:
                    succ = False
                return succ, c_x, cc_x, ccc_x
            for ___ in range(10000//l):
                x_old = cp.deepcopy(x)
                __ = 0
                it = 0
                alpha = 1.0
                phi_ls = 0.01 
                alpha_inc = 1.2
                alpha_dec = 0.5
                for _ in range(l):
                    flag, f_x, ff_x, fff_x = eval_help(x)  
                    if self.ec >= 9990:
                        return x
                    else:
                        self.ec += 2
                    lamb = max(0.0,-np.amin(np.linalg.eig(fff_x)[0])+0.001)
                    damp = lamb * np.eye(n)
                    try:
                        delta = np.linalg.solve(fff_x + damp,-ff_x)
                        if ff_x.T @ delta > 0.0: delta = -ff_x / np.linalg.norm(ff_x) 
                    except:
                        delta = -ff_x / np.linalg.norm(ff_x)
                    flag_n, f_xn, ff_xn, fff_xn = eval_help(x + alpha * delta)
                    if self.ec >= 9990:
                        return x
                    else:
                        self.ec += 2
                    while not flag_n or np.all(np.greater(f_xn, f_x + phi_ls * ff_x.T @ (alpha * delta))):
                        alpha *= alpha_dec
                        flag_n, f_xn, ff_xn, fff_xn = eval_help(x + alpha * delta)
                        if self.ec >= 9990:
                            return x
                        else:
                            self.ec += 2
                        if __ < l: 
                            __ += 1
                        else: 
                            break
                    if np.linalg.norm(alpha * delta) <= 0.0001: 
                        it += 1
                        if it == 5:
                           break 
                    else:
                        it = 0
                    x = x + alpha * delta 
                    alpha = min(alpha_inc * alpha, 1.0)    
                mu = 0.5 * mu
                if np.linalg.norm(x-x_old) <= 0.0001: 
                    _it += 1
                    if _it == 5:
                        break 
                else:
                    _it = 0
        if OT.f in self.ot and OT.sos in self.ot and OT.ineq not in self.ot:
            # x* = argmin f(x) + phi(x)^T @ phi(x)
            # Quasi-Newton
            def eval_help(x):
                phi, J = self.problem.evaluate(x)
                c_x = sum([phi[i][0] if x == OT.f else phi[i].T@phi[i] for i, x in enumerate(self.ot)]) 
                index_r = [i for i, x in enumerate(self.ot) if x == OT.sos] 
                index_f = [i for i, x in enumerate(self.ot) if x == OT.f]
                cc_x = J[index_f][0] + 2.0 * J[index_r].T @ phi[index_r]
                ccc_x = self.problem.getFHessian(x) + 2.0 * J[index_r].T @ J[index_r]
                return c_x, cc_x, ccc_x
            __ = 0
            it = 0
            alpha = 1.0
            phi_ls = 0.01 
            alpha_inc = 1.2
            alpha_dec = 0.5
            for _ in range(1000):
                f_x, ff_x, fff_x = eval_help(x)  
                if self.ec >= 9990:
                    return x
                else:
                    self.ec += 2
                lamb = max(0.0,-np.amin(np.linalg.eig(fff_x)[0])+0.001)
                damp = lamb * np.eye(n)
                try:
                    delta = np.linalg.solve(fff_x + damp,-ff_x)
                    if ff_x.T @ delta > 0.0: delta = -ff_x / np.linalg.norm(ff_x) 
                except:
                    delta = -ff_x / np.linalg.norm(ff_x)
                f_xn, ff_xn, fff_xn = eval_help(x + alpha * delta)
                if self.ec >= 9990:
                    return x
                else:
                    self.ec += 2
                while np.all(np.greater(f_xn, f_x + phi_ls * ff_x.T @ (alpha * delta))):
                    alpha *= alpha_dec
                    f_xn, ff_xn, fff_xn = eval_help(x + alpha * delta)
                    if self.ec >= 9990:
                        return x
                    else:
                        self.ec += 2
                    __ += 1
                if np.linalg.norm(alpha * delta) <= 0.0001: 
                    it += 1
                    if it == 5:
                       break 
                else:
                    it = 0
                x = x + alpha * delta 
                alpha = min(alpha_inc * alpha, 1.0)
        if OT.f in self.ot and OT.sos in self.ot and OT.ineq in self.ot:
            # x* = argmin f(x) + phi(x)^T @ phi(x) s.t. g(x) <= 0
            # Log Barrier w/ Quasi-Newton
            mu = 1.0
            l = 400 
            _it = 0
            def eval_help(x):
                succ = True
                phi, J = self.problem.evaluate(x)
                c_x, cc_x, ccc_x = np.array([]), np.array([]), np.array([])
                index_f = [i for i, x in enumerate(self.ot) if x == OT.f]
                index_s = [i for i, x in enumerate(self.ot) if x == OT.sos] 
                index_g = [i for i, x in enumerate(self.ot) if x == OT.ineq]
                try:
                    c_x = np.sum(phi[index.f]) + phi[index_s].T @ phi[index_s] - mu * sum([ma.log(-phi[i]) for i in index_g]) 
                    cc_x = J[index_f][0] + 2.0 * J[index_s].T @ phi[index_s] - mu * sum([J[i]/phi[i] for i in index_g])
                    ccc_x = self.problem.getFHessian(x) + 2.0 * J[index_s].T @ J[index_s]
                except:
                    succ = False
                return succ, c_x, cc_x, ccc_x
            for ___ in range(10000//l):
                x_old = cp.deepcopy(x)
                __ = 0
                it = 0
                alpha = 1.0
                phi_ls = 0.01 
                alpha_inc = 1.2
                alpha_dec = 0.5
                for _ in range(l):
                    flag, f_x, ff_x, fff_x = eval_help(x)  
                    if self.ec >= 9990:
                        return x
                    else:
                        self.ec += 2
                    lamb = max(0.0,-np.amin(np.linalg.eig(fff_x)[0])+0.001)
                    damp = lamb * np.eye(n)
                    try:
                        delta = np.linalg.solve(fff_x + damp,-ff_x)
                        if ff_x.T @ delta > 0.0: delta = -ff_x / np.linalg.norm(ff_x) 
                    except:
                        delta = -ff_x / np.linalg.norm(ff_x)
                    flag_n, f_xn, ff_xn, fff_xn = eval_help(x + alpha * delta)
                    if self.ec >= 9990:
                        return x
                    else:
                        self.ec += 2
                    while not flag_n or np.all(np.greater(f_xn, f_x + phi_ls * ff_x.T @ (alpha * delta))):
                        alpha *= alpha_dec
                        flag_n, f_xn, ff_xn, fff_xn = eval_help(x + alpha * delta)
                        if self.ec >= 9990:
                            return x
                        else:
                            self.ec += 2
                        if __ < l: 
                            __ += 1
                        else: 
                            break
                    if np.linalg.norm(alpha * delta) <= 0.0001: 
                        it += 1
                        if it == 5:
                           break 
                    else:
                        it = 0
                    x = x + alpha * delta 
                    alpha = min(alpha_inc * alpha, 1.0)    
                mu = 0.5 * mu
                if np.linalg.norm(x-x_old) <= 0.0001: 
                    _it += 1
                    if _it == 5:
                        break 
                else:
                    _it = 0
        return x
      #  if OT.ineq in self.ot:
      #      mu = 1.0
      #      l = 20
      #      _it = 0

      #      for ___ in range(l):         
      #          x_new = cp.deepcopy(x)
      #          d = solve_uncon(x_new,mu,10000/l) - x
      #          x = x + d
      #          mu = 0.5 * mu
      #          if np.linalg.norm(d) <= 0.0001:
      #              _it += 1
      #              if _it == 5:
      #                  break
      #          else:
      #              _it = 0
      #  else:
      #      n = x.shape[0]
      #  return x 

