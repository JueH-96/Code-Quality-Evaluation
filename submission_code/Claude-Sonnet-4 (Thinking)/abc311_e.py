H, W, N = map(int, input().split())

# Create grid to mark holes (1-indexed)
grid = [[0] * (W + 1) for _ in range(H + 1)]
for _ in range(N):
    a, b = map(int, input().split())
    grid[a][b] = 1

# Create 2D prefix sum
prefix = [[0] * (W + 1) for _ in range(H + 1)]
for i in range(1, H + 1):
    for j in range(1, W + 1):
        prefix[i][j] = grid[i][j] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]

count = 0
for n in range(1, min(H, W) + 1):
    for i in range(1, H - n + 2):
        for j in range(1, W - n + 2):
            # Check if square from (i,j) to (i+n-1, j+n-1) has any holes
            holes_in_square = prefix[i+n-1][j+n-1] - prefix[i-1][j+n-1] - prefix[i+n-1][j-1] + prefix[i-1][j-1]
            if holes_in_square == 0:
                count += 1

print(count)