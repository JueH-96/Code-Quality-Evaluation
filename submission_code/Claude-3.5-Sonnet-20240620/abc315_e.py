# YOUR CODE HERE
from collections import deque

def topological_sort(graph, start):
    visited = set()
    stack = []
    order = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)

    dfs(start)

    while stack:
        order.append(stack.pop())

    return order[::-1][1:]  # Exclude the start node (book 1)

N = int(input())
graph = {i: [] for i in range(1, N+1)}
in_degree = {i: 0 for i in range(1, N+1)}

for i in range(1, N+1):
    line = list(map(int, input().split()))
    C_i = line[0]
    for j in range(1, C_i + 1):
        P_ij = line[j]
        graph[i].append(P_ij)
        in_degree[P_ij] += 1

# Find all nodes with in-degree 0 (except book 1)
queue = deque([i for i in range(2, N+1) if in_degree[i] == 0])

# Remove unnecessary nodes
while queue:
    node = queue.popleft()
    for neighbor in graph[node]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0 and neighbor != 1:
            queue.append(neighbor)
    if node != 1:
        graph.pop(node)

# Perform topological sort starting from book 1
result = topological_sort(graph, 1)

print(*result)