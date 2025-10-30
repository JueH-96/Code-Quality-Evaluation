# YOUR CODE HERE
import sys
from collections import defaultdict

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    index = 2
    sets = []
    for _ in range(N):
        A = int(data[index])
        index += 1
        S = set(map(int, data[index:index + A]))
        index += A
        sets.append(S)
    
    parent = list(range(N))
    rank = [0] * N
    contains_1 = set()
    contains_M = set()
    
    for i, S in enumerate(sets):
        if 1 in S:
            contains_1.add(i)
        if M in S:
            contains_M.add(i)
    
    for i in range(N):
        for j in range(i + 1, N):
            if sets[i].intersection(sets[j]):
                union(parent, rank, i, j)
    
    root_1 = set(find(parent, i) for i in contains_1)
    root_M = set(find(parent, i) for i in contains_M)
    
    if root_1.intersection(root_M):
        operations = len(root_1) + len(root_M) - 1
        print(operations)
    else:
        print(-1)

solve()