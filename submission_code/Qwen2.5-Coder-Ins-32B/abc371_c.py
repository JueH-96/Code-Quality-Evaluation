# YOUR CODE HERE
from itertools import permutations

def is_isomorphic(G, H, perm):
    N = len(G)
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in G or (j, i) in G:
                if (perm[i], perm[j]) not in H and (perm[j], perm[i]) not in H:
                    return False
            else:
                if (perm[i], perm[j]) in H or (perm[j], perm[i]) in H:
                    return False
    return True

def min_cost_to_isomorphic(N, G, H, A):
    min_cost = float('inf')
    edges_G = set()
    edges_H = set()
    
    for u, v in G:
        edges_G.add((min(u, v), max(u, v)))
    
    for a, b in H:
        edges_H.add((min(a, b), max(a, b)))
    
    for perm in permutations(range(N)):
        if is_isomorphic(edges_G, edges_H, perm):
            cost = 0
            for i in range(N):
                for j in range(i + 1, N):
                    if (perm[i], perm[j]) in edges_H or (perm[j], perm[i]) in edges_H:
                        if (i, j) not in edges_G:
                            cost += A[i][j - i - 1]
                    else:
                        if (i, j) in edges_G:
                            cost += A[i][j - i - 1]
            min_cost = min(min_cost, cost)
    
    return min_cost

import sys
input = sys.stdin.read().split()
index = 0

N = int(input[index])
index += 1

M_G = int(input[index])
index += 1

G = []
for _ in range(M_G):
    u = int(input[index])
    v = int(input[index + 1])
    G.append((u - 1, v - 1))
    index += 2

M_H = int(input[index])
index += 1

H = []
for _ in range(M_H):
    a = int(input[index])
    b = int(input[index + 1])
    H.append((a - 1, b - 1))
    index += 2

A = []
for i in range(N):
    row = []
    for j in range(N - i - 1):
        row.append(int(input[index]))
        index += 1
    A.append(row)

print(min_cost_to_isomorphic(N, G, H, A))