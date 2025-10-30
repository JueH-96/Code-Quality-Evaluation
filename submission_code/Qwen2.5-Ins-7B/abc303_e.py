# YOUR CODE HERE
from collections import defaultdict

def solve():
    n = int(input())
    edges = [tuple(map(int, input().split())) for _ in range(n-1)]
    graph = defaultdict(set)
    degree = [0] * (n+1)
    
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
        degree[u] += 1
        degree[v] += 1
    
    leaves = [i for i in range(1, n+1) if degree[i] == 1]
    stars = []
    
    while len(leaves) > 1:
        new_leaves = []
        for leaf in leaves:
            parent = next(iter(graph[leaf]))
            degree[parent] -= 1
            if degree[parent] == 1:
                new_leaves.append(parent)
            graph[parent].remove(leaf)
            graph[leaf].remove(parent)
        leaves = new_leaves
    
    for node in range(1, n+1):
        if degree[node] > 1:
            stars.append(len(graph[node]))
    
    stars.sort()
    print(*stars)

solve()