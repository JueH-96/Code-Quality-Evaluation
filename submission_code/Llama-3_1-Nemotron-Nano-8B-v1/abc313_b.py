n, m = map(int, input().split())
adj = [[False] * (n + 1) for _ in range(n + 1)]
for _ in range(m):
    a, b = map(int, input().split())
    adj[a][b] = True

reachable = [row[:] for row in adj]

for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            reachable[i][j] = reachable[i][j] or (reachable[i][k] and reachable[k][j])

maximals = []
for u in range(1, n + 1):
    count = 0
    for v in range(1, n + 1):
        if v != u and reachable[v][u]:
            count += 1
    if count == 0:
        maximals.append(u)

print(maximals[0] if len(maximals) == 1 else -1)