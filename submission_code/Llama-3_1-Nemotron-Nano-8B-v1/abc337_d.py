H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]
min_ops = float('inf')

# Process horizontal windows
for row in grid:
    prefix_x = [0] * (W + 1)
    prefix_dot = [0] * (W + 1)
    for j in range(W):
        prefix_x[j+1] = prefix_x[j] + (1 if row[j] == 'x' else 0)
        prefix_dot[j+1] = prefix_dot[j] + (1 if row[j] == '.' else 0)
    for j in range(W - K + 1):
        end = j + K
        x_count = prefix_x[end] - prefix_x[j]
        if x_count == 0:
            current = prefix_dot[end] - prefix_dot[j]
            if current < min_ops:
                min_ops = current

# Process vertical windows
for j in range(W):
    col = []
    for i in range(H):
        col.append(grid[i][j])
    prefix_x = [0] * (H + 1)
    prefix_dot = [0] * (H + 1)
    for i in range(H):
        prefix_x[i+1] = prefix_x[i] + (1 if col[i] == 'x' else 0)
        prefix_dot[i+1] = prefix_dot[i] + (1 if col[i] == '.' else 0)
    for i in range(H - K + 1):
        end = i + K
        x_count = prefix_x[end] - prefix_x[i]
        if x_count == 0:
            current = prefix_dot[end] - prefix_dot[i]
            if current < min_ops:
                min_ops = current

print(min_ops if min_ops != float('inf') else -1)