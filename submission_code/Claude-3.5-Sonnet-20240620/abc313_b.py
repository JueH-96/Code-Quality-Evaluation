# YOUR CODE HERE
import sys
from collections import defaultdict

def find_strongest(n, edges):
    graph = defaultdict(set)
    indegree = [0] * (n + 1)
    
    for a, b in edges:
        graph[a].add(b)
        indegree[b] += 1
    
    candidates = [i for i in range(1, n + 1) if indegree[i] == 0]
    
    if len(candidates) != 1:
        return -1
    
    strongest = candidates[0]
    visited = set()
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(strongest)
    
    return strongest if len(visited) == n else -1

n, m = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]

print(find_strongest(n, edges))