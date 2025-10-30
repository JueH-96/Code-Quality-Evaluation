# Read input
n, m = map(int, input().split())
adj_matrix = [[False for _ in range(n + 1)] for _ in range(n + 1)]

# Build the graph based on the given information
for _ in range(m):
    a, b = map(int, input().split())
    adj_matrix[a][b] = True

# Apply transitive closure using Floyd-Warshall
for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if adj_matrix[i][k] and adj_matrix[k][j]:
                adj_matrix[i][j] = True

# Find candidates for the strongest programmer
candidates = []
for i in range(1, n + 1):
    is_candidate = True
    for j in range(1, n + 1):
        if i != j and adj_matrix[j][i]:  # If j is stronger than i
            is_candidate = False
            break
    if is_candidate:
        candidates.append(i)

# Output the result
if len(candidates) == 1:
    print(candidates[0])
else:
    print(-1)