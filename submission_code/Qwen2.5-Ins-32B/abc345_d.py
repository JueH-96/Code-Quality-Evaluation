import sys

def can_cover_grid(N, H, W, tiles):
    def dfs(x, y, used):
        if x == W:
            return dfs(0, y + 1, used)
        if y == H:
            return True
        if grid[y][x] != -1:
            return dfs(x + 1, y, used)
        
        for i in range(N):
            if used & (1 << i):
                continue
            for dx, dy in [(tiles[i][0], tiles[i][1]), (tiles[i][1], tiles[i][0])]:
                if x + dx <= W and y + dy <= H:
                    valid = True
                    for ny in range(y, y + dy):
                        for nx in range(x, x + dx):
                            if grid[ny][nx] != -1:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        for ny in range(y, y + dy):
                            for nx in range(x, x + dx):
                                grid[ny][nx] = i
                        if dfs(x + dx, y, used | (1 << i)):
                            return True
                        for ny in range(y, y + dy):
                            for nx in range(x, x + dx):
                                grid[ny][nx] = -1
        return False

    grid = [[-1] * W for _ in range(H)]
    return dfs(0, 0, 0)

N, H, W = map(int, input().split())
tiles = [tuple(map(int, input().split())) for _ in range(N)]

if can_cover_grid(N, H, W, tiles):
    print("Yes")
else:
    print("No")