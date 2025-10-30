n, m = map(int, input().split())

# Initialize the closure matrix
closure = [[False] * (n + 1) for _ in range(n + 1)]

# Read the edges and populate the closure matrix
for _ in range(m):
    a, b = map(int, input().split())
    closure[a][b] = True

# Floyd-Warshall algorithm to compute transitive closure
for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if closure[i][k] and closure[k][j]:
                closure[i][j] = True

# Find all candidates for strongest programmer
candidates = []
for x in range(1, n + 1):
    is_candidate = True
    for y in range(1, n + 1):
        if x != y and closure[y][x]:
            is_candidate = False
            break
    if is_candidate:
        candidates.append(x)

# Determine the output based on the number of candidates
if len(candidates) == 1:
    print(candidates[0])
else:
    print(-1)