n, m = map(int, input().split())

# Initialize reachability matrix
reach = [[False] * (n + 1) for _ in range(n + 1)]

for _ in range(m):
    a, b = map(int, input().split())
    reach[a][b] = True

# Floyd-Warshall algorithm to compute transitive closure
for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if reach[i][k] and reach[k][j]:
                if not reach[i][j]:
                    reach[i][j] = True

candidates = []
for x in range(1, n + 1):
    valid = True
    for y in range(1, n + 1):
        if y == x:
            continue
        if reach[y][x]:
            valid = False
            break
    if valid:
        candidates.append(x)

if len(candidates) == 1:
    print(candidates[0])
else:
    print(-1)