from scipy.optimize import linprog
import numpy as np

N = int(input())
Q = np.array(list(map(int, input().split())))
A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))

c = [-1, -1]
A_ub = np.array([A, B]).T
b_ub = -Q
bounds = [(0, None), (0, None)]

res = linprog(c, A_ub=-A_ub, b_ub=-b_ub, bounds=bounds, method='highs')
print(int(-res.fun))