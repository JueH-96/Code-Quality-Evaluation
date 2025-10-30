from collections import defaultdict

def dfs(node, visited, graph):
    stack = [node]
    visited[node] = True
    degrees = []
    while stack:
        current = stack.pop()
        degrees.append(len(graph[current]))
        for neighbor in graph[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    return degrees

N, M = map(int, input().split())
graph = defaultdict(set)
for _ in range(M):
    A, B = map(int, input().split())
    graph[A].add(B)
    graph[B].add(A)

visited = [False] * (N + 1)
total = 0
for i in range(1, N + 1):
    if not visited[i]:
        degrees = dfs(i, visited, graph)
        degrees.sort()
        for j in range(len(degrees)):
            total += degrees[j] * (len(degrees) - j - 1) - (degrees[j] - 1) * j // 2
print(total)