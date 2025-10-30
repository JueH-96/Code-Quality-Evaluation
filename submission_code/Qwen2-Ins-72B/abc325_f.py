from scipy.optimize import linprog
import numpy as np
def solve():
    N = int(input())
    D = list(map(int, input().split()))
    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())
    L = [L1, L2]
    C = [C1, C2]
    K = [K1, K2]
    if max(L) > sum(D):
        print(-1)
        return
    cost = [-c for c in C]
    A_ub = [[l, 0] for l in L] + [[0, l] for l in L]
    b_ub = D * 2
    A_eq = [[1, 1]]
    b_eq = [1]
    bounds = [(0, k) for k in K]
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        print(int(np.dot(C, res.x) * N))
    else:
        print(-1)
solve()