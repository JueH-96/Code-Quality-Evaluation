# Read the input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

# Find the positions of the cells with s, n, u, k, and e
for i in range(H):
    for j in range(W):
        if grid[i][j] == 's':
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if 0 <= i+di < H and 0 <= j+dj < W and grid[i+di][j+dj] == 'n' and grid[i+2*di][j+2*dj] == 'u' and grid[i+3*di][j+3*dj] == 'k' and grid[i+4*di][j+4*dj] == 'e':
                        print(i+1, j+1)
                        print(i+1+di, j+1+dj)
                        print(i+1+2*di, j+1+2*dj)
                        print(i+1+3*di, j+1+3*dj)
                        print(i+1+4*di, j+1+4*dj)
                        break