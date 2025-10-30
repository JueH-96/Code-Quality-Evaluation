# YOUR CODE HERE
from collections import defaultdict

def dfs(graph, start, parent, degrees):
    stack = [(start, parent)]
    while stack:
        node, parent = stack.pop()
        for neighbor in graph[node]:
            if neighbor != parent:
                degrees[node] += 1
                degrees[neighbor] += 1
                stack.append((neighbor, node))

def count_valid_graphs(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    degrees = defaultdict(int)
    dfs(graph, 1, -1, degrees)
    
    count = 0
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if j not in graph[i]:
                if degrees[i] == 2 and degrees[j] == 2:
                    count += 1
    
    return count

# Read input
n = int(input())
edges = [list(map(int, input().split())) for _ in range(n-1)]

# Solve and print output
result = count_valid_graphs(n, edges)
print(result)