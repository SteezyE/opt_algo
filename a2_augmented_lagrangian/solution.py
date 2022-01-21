import numpy as np
import copy as cp
import math as ma
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverAugmentedLagrangian(NLPSolver):

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
        if OT.ineq in self.ot or OT.eq in self.ot:
            # x* = argmin f(x) + phi(x)^T @ phi(x) s.t. g(x) <= 0 and h(x) = 0
            # Augmented Lagrangian w/ Newton / Quasi-Newton
            l = np.zeros(len(self.ot))
            k = np.zeros(len(self.ot))
            dl = np.zeros(len(self.ot))
            dk = np.zeros(len(self.ot))
            u = 1.0 
            v = 1.0
            w = 400 
            _it = 0
            h_max = 0.0
            g_max = 0.0 
            def eval_help(x):
                phi, J = self.problem.evaluate(x)
                index_r = [i for i, x in enumerate(self.ot) if x == OT.sos] 
                index_f = [i for i, x in enumerate(self.ot) if x == OT.f]
                index_i = [i for i, x in enumerate(self.ot) if x == OT.ineq]
                index_e = [i for i, x in enumerate(self.ot) if x == OT.eq]
                for i in index_i:
                    dl[i] = 2.0 * u * phi[i]
                for i in index_e:
                    dk[i] = 2.0 * v * phi[i]
                c_x = 0.0
                cc_x = np.array([])
                ccc_x = np.array([])
                if len(index_f) > 0:
                    c_x = sum([phi[i] for i in index_f])
                    cc_x = J[index_f][0]
                    ccc_x = self.problem.getFHessian(x)
                if len(index_r) > 0:
                    if len(index_f) > 0:
                        c_x = c_x + phi[index_r].T @ phi[index_r]
                        cc_x = cc_x + 2.0 * J[index_r].T @ phi[index_r]
                        ccc_x = ccc_x + 2.0 * J[index_r].T @ J[index_r]
                    else: 
                        c_x = phi[index_r].T @ phi[index_r]
                        cc_x = 2.0 * J[index_r].T @ phi[index_r]
                        ccc_x = 2.0 * J[index_r].T @ J[index_r]
                if len(index_i) > 0: 
                    g_max = np.amax(phi[index_i])
                    t = np.zeros(J[index_i[0]].shape)
                    for i in index_i:
                        if phi[i] >= 0.0 or l[i] > 0.0: 
                            t = t + phi[i] * J[i]
                    tt = np.zeros(J[index_i[0]].shape)
                    for i in index_i:
                        tt = tt + l[i] * J[i]
                    if len(index_f) > 0 or len(index_r) > 0:
                        c_x = c_x + u * sum([phi[i] * phi[i] if phi[i] >= 0.0 or l[i] > 0.0 else 0.0 for i in index_i]) + sum([l[i] * phi[i] for i in index_i])
                        cc_x = cc_x + 2.0 * u * t + tt                    
                    else:
                        c_x = u * sum([phi[i]*phi[i] if phi[i] >= 0.0 or l[i] > 0.0 else 0.0 for i in index_i]) + sum([l[i] * phi[i] for i in index_i])
                        cc_x = 2.0 * u * t + tt
                if len(index_e) > 0:
                    h_max = np.amax(np.abs(phi[index_e]))
                    t, tt = np.zeros(J[index_e[0]].shape), np.zeros(J[index_e[0]].shape)
                    for i in index_e:
                        t = t + phi[i] * J[i] 
                        tt = tt + k[i] * J[i]
                    if len(index_f)+len(index_r)+len(index_i) > 0: 
                        c_x = c_x + v * sum([phi[i] * phi[i] for i in index_e]) + sum([k[i] * phi[i] for i in index_e])
                        cc_x = cc_x + 2.0 * v * t + tt
                    else:
                        c_x = v * sum([phi[i] * phi[i] for i in index_e]) + sum([k[i] * phi[i] for i in index_e])
                        cc_x = 2.0 * v * t + tt
                return c_x, cc_x, ccc_x
            for ___ in range(10000//w):
                x_old = cp.deepcopy(x)
                __ = 0
                it = 0
                alpha = 1.0
                phi_ls = 0.01 
                alpha_inc = 1.2
                alpha_dec = 0.5
                for _ in range(w):
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
                        if __ < w: 
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
                l = l + dl
                l[l < 0.0] = 0.0
                k = k + dk
                u = 1.2 * u
                v = 1.2 * v
                if max(np.linalg.norm(x-x_old), g_max, h_max) <= 0.0001: 
                    _it += 1
                    if _it == 5:
                        break 
                else:
                    _it = 0
        elif OT.f in self.ot or OT.sos in self.ot:
            # x* = argmin f(x) + phi(x)^T @ phi(x)
            # Newton / Quasi-Newton
            def eval_help(x):
                phi, J = self.problem.evaluate(x)
                c_x = sum([phi[i][0] if x == OT.f else phi[i].T@phi[i] for i, x in enumerate(self.ot)]) 
                index_r = [i for i, x in enumerate(self.ot) if x == OT.sos] 
                index_f = [i for i, x in enumerate(self.ot) if x == OT.f]
                cc_x = np.array([])
                ccc_x = np.array([])
                if len(index_r) > 0:
                    cc_x = 2.0 * J[index_r].T @ phi[index_r]
                    ccc_x = 2.0 * J[index_r].T @ J[index_r]
                if len(index_f) > 0:
                    if len(index_r) > 0:
                        cc_x = cc_x + J[index_f][0]
                        ccc_x = ccc_x + self.problem.getFHessian(x)
                    else:
                        cc_x = J[index_f][0]
                        ccc_x = self.problem.getFHessian(x)
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
        return x
