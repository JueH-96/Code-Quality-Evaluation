# Read input
H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

min_operations = float('inf')

# Check all horizontal sequences
for i in range(H):
    row = grid[i]
    if len(row) < K:
        continue
    for j in range(len(row) - K + 1):
        window = row[j:j+K]
        if 'x' in window:
            continue
        current = window.count('.')
        if current < min_operations:
            min_operations = current

# Check all vertical sequences
for j in range(W):
    column = [grid[i][j] for i in range(H)]
    for i in range(len(column) - K + 1):
        window = column[i:i+K]
        if 'x' in window:
            continue
        current = window.count('.')
        if current < min_operations:
            min_operations = current

if min_operations != float('inf'):
    print(min_operations)
else:
    print(-1)