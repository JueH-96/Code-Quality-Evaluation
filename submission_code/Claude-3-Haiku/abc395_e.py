from collections import deque

def min_cost_to_reach_n(n, m, x, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u-1].append(v-1)

    queue = deque([(0, 0)])  # (vertex, cost)
    visited = set([0])
    min_cost = float('inf')

    while queue:
        vertex, cost = queue.popleft()
        if vertex == n-1:
            min_cost = min(min_cost, cost)
            continue

        # Move along a directed edge
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, cost + 1))

        # Reverse all edges
        for i in range(n):
            for j in graph[i]:
                if j == vertex:
                    graph[j].remove(i)
                    graph[i].append(j)
        visited.add(vertex)
        queue.append((vertex, cost + x))

    return min_cost

# Read input
n, m, x = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# Solve and print the answer
print(min_cost_to_reach_n(n, m, x, edges))