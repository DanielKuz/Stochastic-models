import math
import numpy as np
import matplotlib.pyplot as plt
import mealpy
import os
import pandas as pd
from functools import reduce
import operator






class Hw2:
    def __init__(self, func=None, d=1, epoch=200, gen=None):
        self.func = func
        self.d = d
        self.epoch = epoch
        self.gen = gen

    def rosenbrock(self, x, y):
        res = (x-1)**2 + 100*((y-x**2)**2)
        return res

    def rosenbrock18(self, x):
        x = np.asarray_chkfinite(x)
        x0 = x[:-1]
        x1 = x[1:]
        return sum((1 - x0) ** 2) + 100 * sum((x1 - x0 ** 2) ** 2)

    def plot(self, func, x, y, z):
        # Initialize figure
        figRos = plt.figure(figsize=(12, 7))
        axRos = plt.axes(projection='3d')

        # Evaluate function
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-0.5, 1.5, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = self.rosenbrock(X, Y)

        # Plot
        surf = axRos.plot_surface(X, Y, Z, cmap='gist_heat_r', linewidth=0, antialiased=False)
        axRos.set_zlim(0, 200)
        figRos.colorbar(surf, shrink=0.5, aspect=10)

        # Plot points
        axRos.scatter3D(x, y, z, c=z, cmap='autumn')  # TODO change colors

        plt.show()

    def HC(self):
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.math_based.HC.BaseHC(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness

    def GWO(self):
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.swarm_based.GWO.BaseGWO(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness

    def SSA(self):
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.swarm_based.SSA.BaseSSA(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness

    def SSO(self):
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.swarm_based.SSO.BaseSSO(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness

    def HHO(self):
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.swarm_based.HHO.BaseHHO(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness

    def PSO(self):
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.swarm_based.PSO.BasePSO(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness

    # FIXME
    def SPC(self):   # FIXME
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.swarm_based.SSpiderO.BaseSSpiderO(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness  # FIXEM

    def EHO(self):
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.swarm_based.EHO.BaseEHO(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness

    def WOH(self):
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.swarm_based.WOA.BaseWOA(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness

    def SA(self):
        problem_dict = {
            "fit_func": self.func,
            "lb": [-10] * self.d,
            "ub": [10] * self.d,
            "minmax": "min",
            "generate_position": self.gen,
            "verbose": True}

        # Create model
        model = mealpy.physics_based.SA.BaseSA(problem_dict, epoch=self.epoch)
        best_position, best_fitness = model.solve()
        # x, y, z = [], [], []
        # for point in model.history.list_global_best:
        #     x.append(point[0][0])
        #     y.append(point[0][1])
        #     z.append(self.rosenbrock(point[0][0], point[0][1]))
        # model.history.save_global_objectives_chart()
        return best_fitness


def zakharov(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1., n + 1)
    s2 = sum(j * x) / 2
    return sum(x ** 2) + s2 ** 2 + s2 ** 4


def dixonprice(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(2, n + 1)
    x2 = 2 * x ** 2
    return sum(j *(x2[1:] - x[:-1])** 2) + (x[0] - 1)** 2


def opt_func(solution):
    def c1(x):
        return 1 - ((x[1])**3)*(x[2])/(71785*(x[0]**4))
    def c2(x):
        return (4*(x[1]**2)-x[0]*x[1])/(12566*(x[1]*(x[0]**3)-(x[0]**4))) + 1/(5108*(x[0]**2))
    def c3(x):
        return 1 - (140.45*x[0])/((x[1]**2)*x[2])
    def c4(x):
        return (x[0]+x[1])/1.5 - 1
    def violate(value):
        return 0 if value <= 0 else value

    fx = (solution[2]+2)*(solution[1]*(solution[0]**2))
    fx += violate(c1(solution)) + violate(c2(solution)) + violate(c3(solution)) + violate(c4(solution))

    return fx


# [-100, 100]
def f1(x):
    x = np.asarray_chkfinite(x)
    return sum(x ** 2)


# [-100, 100]
def f2(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(0, n)
    res = sum(abs(x)**(j+2))
    return res


# [-65, 65]
def f3(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    i = np.arange(0, n)
    res = 0
    for j in range(len(i)):
        res += sum(x[:j+1]**2)
    return res


# [-100, 100]
def f6(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    res = 0
    for j in range(n):
        res += (sum(x[:j+1]))**2
    return res


# [-10, 10]
def f11(x):
    x = np.asarray_chkfinite(x)
    return sum(abs(x))+reduce(operator.mul, abs(x), 1)


# [-100, 100]
def f12(x):
    x = np.asarray_chkfinite(x)
    return sum((abs(x)+0.5)**2)


# [-600, 600]
def f13(x):
    x = np.asarray_chkfinite(x)
    res = 1
    for i in range(len(x)):
        res *= np.cos(x[i]/np.sqrt(i+1))
    return sum((x**2)/4000) - res + 1


# [-d^2, d^2]
def f14(x):
    x = np.asarray_chkfinite(x)
    return sum((x - 1) **2) - sum(x[:-1] * x[1:])


# [-5.12, 5.12]
def f15(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum(x**2 - 10 * np.cos(2 * np.pi * x))


# [-10, 10]
def f24(x):
    x = np.asarray_chkfinite(x)
    return sum(abs(x*np.sin(x)+0.1*x))


# [-100, 100]
def f27(x):
    x = np.asarray_chkfinite(x)
    return 1 - np.cos(2*np.pi*sum(x)) + 0.1*sum(x**2)


def generate_position(lb=None, ub=None):
    return np.array([5, 5, 5])


def generate_position_rand(lb=None, ub=None):
    n = 5
    np.random.seed(8)
    return np.random.uniform(0, 1, n)

def main(seif):
    if seif == 'alef':
        Ini = Hw2(opt_func, 5, 20, generate_position)
        algs = [Ini.HC, Ini.GWO, Ini.SSA, Ini.SSO]
        s_algs = ["EHO", "SA", "GWO", "HHO", "PSO", "WOH", "HC"]
        output = {}
        i = 0
        for a in algs:
            res = []
            for _ in range(1):
                r = a()
                res.append(r)
            # Calculate std, mean
            output[s_algs[i]] = [np.mean(res), np.std(res)]
            i += 1
        print(output)

    else:  # Seif b
        funcs = [#[f1, -100, 100],
                 #[f2, -100, 100],
                #  [f3, -65, 65],
                 # [f6, -100, 100],
                #  [f11, -10, 10],
                 #  [f12, -100, 100],
                [f13, -600, 600], # CHANGE EVERYTIME YOU SWITCH A FUNCTION ********change this line**
                #  [f14, -25, 25],
                #  [f15,-5.12, 5.12],
                #  [f24, -10, 10],
                #  [f27, -100, 100]
                 ]
        final = {}
        for indx, f in enumerate(funcs):
            s_algs = ["EHO", "SA", "GWO", "HHO", "PSO", "WOH", "HC"]
            dims = [5, 10, 50, 100]
            output_for_f = {}
            for d in dims:
                # if f[0] == funcs[7][0]:  # For f14 only
                #     f[1] = -1*(d**2)
                #     f[2] = d**2

                Ini = Hw2(f[0], d, 200, lambda x, y: np.random.uniform(f[1], f[2], d))
                algs = [Ini.EHO, Ini.SA, Ini.GWO, Ini.HHO, Ini.PSO, Ini.WOH, Ini.HC]
                output = {}
                i = 0
                for a in algs:
                    np.random.seed(8)
                    res = []
                    for _ in range(30):
                        r = a()
                        res.append(r)
                    # Calculate std, mean
                    output[s_algs[i]] = output.get(s_algs[i], []) + [np.mean(res)] + [np.std(res)]
                    i += 1
                output_for_f[d] = output
            final[indx] = output_for_f
        return final

def res_to_csv(data):
    final_df = pd.DataFrame()
    for func, vals in data.items():
        df = pd.DataFrame()
        for d, algs in vals.items():
            tmpDf = pd.DataFrame({d: list(algs.values())})
            tmpDf.index = ["EHO", "SA", "GWO", "HHO", "PSO", "WOH", "HC"]
            df = pd.concat([df, tmpDf], axis=1)

        final_df = final_df.append(df)

    current_func = 'f13'  # CHANGE EVERYTIME YOU SWITCH A FUNCTION ********change this line**
    final_df.to_csv(f'{current_func}.csv', header=True, index=True)


if __name__ == '__main__':
    data = main('bet')
    res_to_csv(data)