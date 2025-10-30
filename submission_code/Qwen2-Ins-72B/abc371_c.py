from itertools import permutations
from collections import defaultdict

def calc_cost(G, H, A, perm):
    cost = 0
    H_perm = [(perm[a-1]+1, perm[b-1]+1) for a, b in H]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            if (i, j) in G and (i, j) not in H_perm:
                cost += A[i-1][j-1]
            elif (i, j) not in G and (i, j) in H_perm:
                cost += A[i-1][j-1]
    return cost

N = int(input())
M_G = int(input())
G = [tuple(map(int, input().split())) for _ in range(M_G)]
M_H = int(input())
H = [tuple(map(int, input().split())) for _ in range(M_H)]
A = [list(map(int, input().split())) for _ in range(N-1)]
for i in range(N-1):
    A[i].insert(i, 0)

min_cost = float('inf')
for perm in permutations(range(1, N+1)):
    cost = calc_cost(G, H, A, perm)
    min_cost = min(min_cost, cost)

print(min_cost)