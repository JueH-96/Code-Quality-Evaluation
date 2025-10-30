import sys

def count_sensors(H, W, grid):
    def dfs(i, j):
        if 0 <= i < H and 0 <= j < W and grid[i][j] == '#':
            grid[i][j] = '.'
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    dfs(i+dx, j+dy)

    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                dfs(i, j)
                count += 1
    return count

H, W = map(int, sys.stdin.readline().split())
grid = [list(sys.stdin.readline().rstrip()) for _ in range(H)]
print(count_sensors(H, W, grid))