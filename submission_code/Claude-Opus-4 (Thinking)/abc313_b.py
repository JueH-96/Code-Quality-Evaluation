# YOUR CODE HERE
N, M = map(int, input().split())

# Create adjacency matrix for the superiority graph
# superior[i][j] = True means person i is stronger than person j
superior = [[False] * (N + 1) for _ in range(N + 1)]

# Read the given relations
for _ in range(M):
    a, b = map(int, input().split())
    superior[a][b] = True

# Compute transitive closure using Floyd-Warshall
for k in range(1, N + 1):
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if superior[i][k] and superior[k][j]:
                superior[i][j] = True

# Find potential strongest programmers
potential_strongest = []

for i in range(1, N + 1):
    is_strongest = True
    for j in range(1, N + 1):
        if i != j and superior[j][i]:
            is_strongest = False
            break
    if is_strongest:
        potential_strongest.append(i)

# If exactly one strongest programmer, print them; otherwise -1
if len(potential_strongest) == 1:
    print(potential_strongest[0])
else:
    print(-1)