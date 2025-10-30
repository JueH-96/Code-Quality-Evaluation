# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            print(i+1, j+1)
            break