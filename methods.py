import time
import numpy as np
from numpy.linalg import norm


def gradf_inexact(w, gradf, Delta=0, dtype=1, v=None):
    n = len(w)
    g = gradf(w)
    xi = np.random.normal(0, 1, (n,))
    if dtype == 1:
        xi /= np.linalg.norm(xi)
    elif dtype == 2:
        xi /= n
    elif dtype == 3:
        xi = -g.copy()
        xi /= np.linalg.norm(xi)
    elif dtype == 4:
        xi = v / np.linalg.norm(v)
    assert np.linalg.norm(xi) <= 1 + 1e-9, "xi has no 1 norm"
    return g + Delta * xi


class GradientDescent:
    def __init__(self, StepSizeChoice, return_history=True, name=None, save_iter=1):
        self.name = name
        self.StepSizeChoice = StepSizeChoice
        self.return_history = return_history
        self.history = []
        self.save_iter = save_iter

    def solve(self, x0, f, gradf, tol=1e-3, max_iter=10000):
        self.history = [(x0, time.time())]
        x = x0.copy()
        k = 0
        x_prev = None
        while x_prev is None or np.linalg.norm(gradf(x)) > tol:
            h = -gradf(x)
            alpha = self.StepSizeChoice(x, h, k, gradf, f)
            x_prev, x = x, x + alpha * h
            if self.return_history and k % self.save_iter == 0:
                self.history.append((x, time.time()))
            if k >= max_iter:
                break
            k += 1
        return x


def parse_logs(xhistory, ret_time=False, funcx=None):
    values = [funcx(x) for x, _ in xhistory]
    if ret_time:
        times = [t for _, t in xhistory]
        times = [times[ind] - times[0] for ind, t in enumerate(times)]
    else:
        times = [i for i in range(len(xhistory))]
    return times, np.array(values)


class StepSize:
    def __call__(self, x, h, k, gradf, f, *args, **kwargs):
        pass


class ConstantStepSize(StepSize):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x, h, k, gradf, f, *args, **kwargs):
        return self.alpha


class AdaptiveL(StepSize):
    def __init__(self, L0=1, Delta=0, Lmin=0):
        self.L = L0
        self.Lmin = Lmin
        self.Delta = Delta

    def __call__(self, x, h, k, gradf, f, *args, **kwargs):
        L, Delta = self.L, self.Delta
        L = max(L / 2, self.Lmin)
        xnew = x + 1 / (2 * L) * h
        normh = norm(h)
        fx = f(x)
        while f(xnew) > fx - normh ** 2 / (2 * L) + 1 / (8 * L) * normh ** 2 + Delta * normh / (2 * L):
            L *= 2
            xnew = x + 1 / (2 * L) * h
        self.L = L
        return 1 / (2 * L)


class AdaptiveNoiseGD:
    def __init__(self, StepSizeChoice, return_history=True, name=None, save_iter=1, alpha=2):
        self.name = name
        self.StepSizeChoice = StepSizeChoice
        self.return_history = return_history
        self.history = []
        self.deltahistory = []
        self.Lhistory = []
        self.save_iter = save_iter
        self.alpha = alpha

    def solve(self, x0, f, gradf, max_iter=10000):
        self.history = [(x0, time.time())]
        x = x0.copy()
        k = 0
        x_prev = None
        while x_prev is None or np.linalg.norm(gradf(x)) > self.alpha * self.StepSizeChoice.maxdelta:
            h = -gradf(x)
            alpha = self.StepSizeChoice(x, h, k, gradf, f)
            x_prev, x = x, x + alpha * h
            if self.return_history and k % self.save_iter == 0:
                self.history.append((x, time.time()))
                self.deltahistory.append(self.StepSizeChoice.Delta)
                self.Lhistory.append(self.StepSizeChoice.Delta)
            if k >= max_iter:
                break
            k += 1
        return x


class AdaptiveLdelta(StepSize):
    def __init__(self, L0=1, Delta0=1e-4, mindelta=1e-4, Lmin=0, fstar=0, mu=1):
        self.L = L0
        self.Lmin = Lmin
        self.Delta = Delta0
        self.maxdelta = 0
        self.mindelta = mindelta
        self.fstar = fstar
        self.mu = mu

    def __call__(self, x, h, k, gradf, f, *args, **kwargs):
        L = self.L
        L = max(L / 2, self.Lmin)
        xnew = x + 1 / (2 * L) * h
        normh = norm(h)
        fx = f(x)
        Delta1 = self.mu * (fx - self.fstar) - normh ** 2
        if Delta1 < 0:
            Delta1 = 0
        Delta1 = np.sqrt(Delta1)
        Delta = max(self.Delta, np.sqrt(Delta1))
        #print(self.Delta)

        while f(xnew) > fx - normh ** 2 / (2 * L) + 1 / (8 * L) * normh ** 2 + Delta * normh / (2 * L):
            L *= 2
            Delta *= 2
            #print("\tWhile", Delta, L)
            xnew = x + 1 / (2 * L) * h

        Delta2 = (f(xnew) - fx + normh ** 2 / (2 * L) - 1 / (8 * L) * normh ** 2) / (normh / (2 * L))
        #print(Delta2, Delta1, self.maxdelta, normh, L)
        Delta = max(Delta1, Delta2, self.maxdelta, self.mindelta)
        self.maxdelta = max(self.maxdelta, Delta)
        self.Delta = Delta

        L = max(L / 2, self.Lmin)
        xnew = x + 1 / (2 * L) * h
        while f(xnew) <= fx - normh ** 2 / (2 * L) + 1 / (8 * L) * normh ** 2 + Delta * normh / (
                2 * L) and L > self.Lmin:
            L = max(L / 2, self.Lmin)
            xnew = x + 1 / (2 * L) * h
        if f(xnew) > fx - normh ** 2 / (2 * L) + 1 / (8 * L) * normh ** 2 + Delta * normh / (
                2 * L):
            L *= 2
        self.L = L
        return 1 / (2 * L)

