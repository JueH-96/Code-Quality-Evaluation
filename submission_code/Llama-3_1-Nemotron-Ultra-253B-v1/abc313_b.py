n, m = map(int, input().split())
reachable = [[False] * (n + 1) for _ in range(n + 1)]

for i in range(n + 1):
    reachable[i][i] = True

for _ in range(m):
    a, b = map(int, input().split())
    reachable[a][b] = True

for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if reachable[i][k] and reachable[k][j]:
                reachable[i][j] = True

candidates = []
for x in range(1, n + 1):
    is_candidate = True
    for y in range(1, n + 1):
        if y != x and reachable[y][x]:
            is_candidate = False
            break
    if is_candidate:
        candidates.append(x)

print(candidates[0] if len(candidates) == 1 else -1)