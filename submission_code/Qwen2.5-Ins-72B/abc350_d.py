import sys
from collections import defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    edges = [(int(data[2*i+2]), int(data[2*i+3])) for i in range(M)]
    
    if M == 0:
        print(0)
        return
    
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    
    visited = [False] * (N + 1)
    components = []
    
    def dfs(node, component):
        stack = [node]
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                component.append(current)
                for neighbor in graph[current]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
    
    for i in range(1, N + 1):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    result = 0
    for component in components:
        if len(component) < 3:
            continue
        degree_sum = sum(len(graph[node]) for node in component)
        result += (len(component) - 1) * len(component) - degree_sum
    
    print(result // 2)

solve()