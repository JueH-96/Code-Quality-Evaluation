H, W, K = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())

min_ops = float('inf')

# Check horizontal sequences
for i in range(H):
    for j in range(W - K + 1):
        ops = 0
        valid = True
        for k in range(K):
            if grid[i][j + k] == 'x':
                valid = False
                break
            elif grid[i][j + k] == '.':
                ops += 1
        if valid:
            min_ops = min(min_ops, ops)

# Check vertical sequences
for i in range(H - K + 1):
    for j in range(W):
        ops = 0
        valid = True
        for k in range(K):
            if grid[i + k][j] == 'x':
                valid = False
                break
            elif grid[i + k][j] == '.':
                ops += 1
        if valid:
            min_ops = min(min_ops, ops)

if min_ops == float('inf'):
    print(-1)
else:
    print(min_ops)