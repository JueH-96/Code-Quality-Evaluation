# YOUR CODE HERE
n = int(input())
edges = []
graph = [[] for _ in range(n)]
degrees = [0] * n
for _ in range(n - 1):
    u, v = map(int, input().split())
    u -= 1
    v -= 1
    edges.append((u, v))
    graph[u].append(v)
    graph[v].append(u)
    degrees[u] += 1
    degrees[v] += 1

leaves = []
for i in range(n):
    if degrees[i] == 1:
        leaves.append(i)

levels = []
while leaves:
    level = 0
    next_leaves = []
    for leaf in leaves:
        level += 1
        neighbors = graph[leaf]
        for neighbor in neighbors:
            degrees[neighbor] -= 1
            if degrees[neighbor] == 1:
                next_leaves.append(neighbor)
    levels.append(level)
    leaves = next_leaves

levels.sort()
print(*levels)