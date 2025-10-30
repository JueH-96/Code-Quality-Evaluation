H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Precompute prefix_x and prefix_dot for each row
prefix_x = []
prefix_dot = []
for row in grid:
    px = [0] * (W + 1)
    pd = [0] * (W + 1)
    for j in range(W):
        if row[j] == 'x':
            px[j+1] = px[j] + 1
            pd[j+1] = pd[j]
        else:
            px[j+1] = px[j]
            pd[j+1] = pd[j] + (1 if row[j] == '.' else 0)
    prefix_x.append(px)
    prefix_dot.append(pd)

# Precompute prefix_x_col and prefix_dot_col for each column
prefix_x_col = []
prefix_dot_col = []
for j in range(W):
    px = [0] * (H + 1)
    pd = [0] * (H + 1)
    for i in range(H):
        cell = grid[i][j]
        if cell == 'x':
            px[i+1] = px[i] + 1
            pd[i+1] = pd[i]
        else:
            px[i+1] = px[i]
            pd[i+1] = pd[i] + (1 if cell == '.' else 0)
    prefix_x_col.append(px)
    prefix_dot_col.append(pd)

min_ops = float('inf')

# Check horizontal runs
for i in range(H):
    for j in range(W - K + 1):
        end = j + K
        x = prefix_x[i][end] - prefix_x[i][j]
        if x == 0:
            ops = prefix_dot[i][end] - prefix_dot[i][j]
            if ops < min_ops:
                min_ops = ops

# Check vertical runs
for j in range(W):
    for i in range(H - K + 1):
        end = i + K
        x = prefix_x_col[j][end] - prefix_x_col[j][i]
        if x == 0:
            ops = prefix_dot_col[j][end] - prefix_dot_col[j][i]
            if ops < min_ops:
                min_ops = ops

if min_ops != float('inf'):
    print(min_ops)
else:
    print(-1)