# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            continue
        if i > 0 and grid[i-1][j] == '.':
            continue
        if i < H-1 and grid[i+1][j] == '.':
            continue
        if j > 0 and grid[i][j-1] == '.':
            continue
        if j < W-1 and grid[i][j+1] == '.':
            continue
        print(i+1, j+1)
        break