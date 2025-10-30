# YOUR CODE HERE
from collections import defaultdict
import sys

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def solve():
    N, M = map(int, input().split())
    parent = list(range(M + 1))
    rank = [0] * (M + 1)
    
    graph = defaultdict(set)
    for _ in range(N):
        A = int(input())
        S = list(map(int, input().split()))
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                graph[S[i]].add(S[j])
                graph[S[j]].add(S[i])
    
    queue = [(1, 0)]
    visited = set()
    
    while queue:
        node, ops = queue.pop(0)
        if node == M:
            print(ops)
            return
        
        if node in visited:
            continue
        visited.add(node)
        
        for neighbor in graph[node]:
            if find(parent, node) != find(parent, neighbor):
                union(parent, rank, node, neighbor)
                queue.append((neighbor, ops + 1))
    
    print(-1)

solve()