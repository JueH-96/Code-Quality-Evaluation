n, m = map(int, input().split())

# stronger[i][j] = True means person i+1 is stronger than person j+1
stronger = [[False] * n for _ in range(n)]

for _ in range(m):
    a, b = map(int, input().split())
    stronger[a-1][b-1] = True

# Apply transitive closure using Floyd-Warshall
for k in range(n):
    for i in range(n):
        for j in range(n):
            if stronger[i][k] and stronger[k][j]:
                stronger[i][j] = True

# Find all potential strongest programmers (those not beaten by anyone)
candidates = []
for i in range(n):
    beaten_by_someone = False
    for j in range(n):
        if i != j and stronger[j][i]:
            beaten_by_someone = True
            break
    if not beaten_by_someone:
        candidates.append(i + 1)

# Output result
if len(candidates) == 1:
    print(candidates[0])
else:
    print(-1)