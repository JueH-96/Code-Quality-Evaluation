# YOUR CODE HERE
H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

min_ops = float('inf')

# Check horizontal sequences
for i in range(H):
    row = grid[i]
    for j in range(W - K + 1):
        segment = row[j:j+K]
        x_count = segment.count('x')
        if x_count > 0:
            continue
        o_count = segment.count('o')
        needed = K - o_count
        if needed < min_ops:
            min_ops = needed

# Check vertical sequences
for j in range(W):
    for i in range(H - K + 1):
        segment = [grid[i+k][j] for k in range(K)]
        x_count = segment.count('x')
        if x_count > 0:
            continue
        o_count = segment.count('o')
        needed = K - o_count
        if needed < min_ops:
            min_ops = needed

if min_ops != float('inf'):
    print(min_ops)
else:
    print(-1)