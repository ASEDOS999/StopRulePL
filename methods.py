import time
import numpy as np
from numpy.linalg import norm


def gradf_inexact(w, gradf, Delta=0, dtype=1, v=None):
    n = len(w)
    g = gradf(w)
    xi = np.random.normal(0, 1, (n,))
    if dtype==1:
        xi /= np.linalg.norm(xi)
    elif dtype==2:
        xi /= n
    elif dtype == 3:
        xi = -g.copy()
        xi /= np.linalg.norm(xi)
    elif dtype == 4:
        xi = v / np.linalg.norm(v)
    assert np.linalg.norm(xi)<=1+1e-9, "xi has no 1 norm"
    return g + Delta * xi


class GradientDescent:
    def __init__(self, StepSizeChoice, return_history=True, name=None, save_iter=1):
        self.name = name
        self.StepSizeChoice = StepSizeChoice
        self.return_history = return_history
        self.history = []
        self.save_iter = save_iter
    
    def __call__(self, x0, f, gradf, N):
        self.history = [(x0, time.time())]
        x = x0.copy()
        for k in range(N):
            h = -gradf(x)
            alpha = self.StepSizeChoice(x, h, k, gradf, f)
            x = x + alpha * h
            if self.return_history:
                self.history.append((x, time.time()))
        return x
    
    def solve(self, x0, f, gradf, tol=1e-3, max_iter=10000):
        self.history = [(x0, time.time())]
        x = x0.copy()
        k = 0
        x_prev = None
        while x_prev is None or np.linalg.norm(gradf(x)) > tol: 
            h = -gradf(x)
            alpha = self.StepSizeChoice(x, h, k, gradf, f)
            x_prev, x = x, x + alpha * h
            if self.return_history and k%self.save_iter==0:
                self.history.append((x, time.time()))
            if k >= max_iter:
                break
            k += 1
        return x


def parse_logs(xhistory, ret_time=False, funcx=None):
    values = [funcx(x) for x, _ in xhistory]
    if ret_time:
        times = [t for _, t in xhistory]
        times = [times[ind]-times[0] for ind, t in enumerate(times)]
    else:
        times = [i for i in range(len(xhistory))]
    return times, np.array(values)


class StepSize:
    def __call__(self, x, h, k, *args, **kwargs):
        pass


class ConstantStepSize(StepSize):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, x, h, k, *args, **kwargs):
        return self.alpha


class AdaptiveL(StepSize):
    def __init__(self, L0, Delta):
        self.L = L0
        self.Delta = Delta
    
    def __call__(self, x, h, k, *args, **kwargs):
        L, Delta = self.L, self.Delta
        xnew = x + 1/(2*L) * h
        while f(xnew) > f(x) + h.dot(xnew-x) + L/2*norm(xnew-x)**2 + Delta * norm(xnew-x):
            L *= 2
            xnew = x + 1/(2*L) * h
        self.L = L/2
        return 1/(2*L)
