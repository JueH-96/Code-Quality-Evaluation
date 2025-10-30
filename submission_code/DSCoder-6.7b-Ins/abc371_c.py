import sys
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def read_input():
    N = int(sys.stdin.readline().strip())
    M_G = int(sys.stdin.readline().strip())
    edges_G = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(M_G)]
    M_H = int(sys.stdin.readline().strip())
    edges_H = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(M_H)]
    A = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
    return N, M_G, edges_G, M_H, edges_H, A

def solve():
    N, M_G, edges_G, M_H, edges_H, A = read_input()
    G = defaultdict(list)
    H = defaultdict(list)
    for u, v in edges_G:
        G[u-1].append(v-1)
        G[v-1].append(u-1)
    for u, v in edges_H:
        H[u-1].append(v-1)
        H[v-1].append(u-1)
    cost = [[A[i][j] for j in range(N)] for i in range(N)]
    row_ind, col_ind = linear_sum_assignment(cost)
    min_cost = sum(cost[i][j] for i, j in zip(row_ind, col_ind))
    return min_cost

print(solve())