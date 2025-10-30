from collections import defaultdict, deque

N = int(input())
A = list(map(int, input().split()))

G = defaultdict(list)
for i, a in enumerate(A):
    G[a-1].append(i)

# topological sort
order = []
visited = [False] * N
for i in range(N):
    if visited[i]:
        continue
    stack = [i]
    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        order.append(v)
        for u in G[v]:
            stack.append(u)

# count the number of vertices in each strongly connected component
scc = defaultdict(int)
for v in reversed(order):
    if scc[v] > 0:
        continue
    stack = [v]
    while stack:
        v = stack.pop()
        if scc[v] > 0:
            continue
        scc[v] = 1
        for u in G[v]:
            stack.append(u)

# count the number of pairs of vertices (u, v) such that vertex v is reachable from vertex u
ans = 0
for v in range(N):
    for u in G[v]:
        ans += scc[u]
print(ans)