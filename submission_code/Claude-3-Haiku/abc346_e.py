from collections import defaultdict

# Read input
H, W, M = map(int, input().split())
operations = []
for _ in range(M):
    T, A, X = map(int, input().split())
    operations.append((T, A, X))

# Apply operations
grid = [[0] * W for _ in range(H)]
for T, A, X in operations:
    if T == 1:
        for j in range(W):
            grid[A-1][j] = X
    else:
        for i in range(H):
            grid[i][A-1] = X

# Count colors
color_counts = defaultdict(int)
for row in grid:
    for cell in row:
        color_counts[cell] += 1

# Print output
print(len(color_counts))
for color, count in sorted(color_counts.items()):
    print(color, count)