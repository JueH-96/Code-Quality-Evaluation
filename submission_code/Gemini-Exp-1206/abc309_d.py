def bfs(graph, start):
    n = len(graph)
    dist = [-1] * n
    dist[start] = 0
    q = [start]
    while q:
        v = q.pop(0)
        for u in graph[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist

n1, n2, m = map(int, input().split())
graph = [[] for _ in range(n1 + n2)]
for _ in range(m):
    a, b = map(int, input().split())
    a -= 1
    b -= 1
    graph[a].append(b)
    graph[b].append(a)

dist1 = bfs(graph, 0)
dist2 = bfs(graph, n1 + n2 - 1)

max_dist1 = 0
for i in range(n1):
    if dist1[i] != -1:
        max_dist1 = max(max_dist1, dist1[i])

max_dist2 = 0
for i in range(n1, n1 + n2):
    if dist2[i] != -1:
        max_dist2 = max(max_dist2, dist2[i])

print(max_dist1 + max_dist2 + 1)