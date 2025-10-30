# YOUR CODE HERE

import sys
from collections import defaultdict
from itertools import permutations

def read_input():
    N = int(input())
    M_G = int(input())
    G = defaultdict(list)
    for _ in range(M_G):
        u, v = map(int, input().split())
        G[u].append(v)
        G[v].append(u)
    M_H = int(input())
    H = defaultdict(list)
    for _ in range(M_H):
        a, b = map(int, input().split())
        H[a].append(b)
        H[b].append(a)
    A = []
    for _ in range(N):
        A.append(list(map(int, input().split())))
    return N, G, H, A

def is_isomorphic(G, H, P):
    for u in G:
        for v in G[u]:
            if P[v] not in H[P[u]]:
                return False
    return True

def solve():
    N, G, H, A = read_input()
    min_cost = float('inf')
    for P in permutations(range(1, N+1)):
        cost = 0
        for i in range(N):
            for j in range(i+1, N):
                if P[j] not in H[P[i]]:
                    cost += A[i][j-i-1]
        if is_isomorphic(G, H, P):
            min_cost = min(min_cost, cost)
    print(min_cost)

solve()