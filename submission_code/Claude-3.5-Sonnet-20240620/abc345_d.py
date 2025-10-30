# YOUR CODE HERE
import sys
from itertools import permutations, product

def can_cover_grid(H, W, tiles):
    def dfs(grid, tiles, i, j):
        if i == H:
            return True
        if j == W:
            return dfs(grid, tiles, i + 1, 0)
        if grid[i][j]:
            return dfs(grid, tiles, i, j + 1)
        
        for k, (a, b) in enumerate(tiles):
            if i + a <= H and j + b <= W and all(not grid[y][x] for y, x in product(range(i, i + a), range(j, j + b))):
                for y, x in product(range(i, i + a), range(j, j + b)):
                    grid[y][x] = True
                if dfs(grid, tiles[:k] + tiles[k+1:], i, j + b):
                    return True
                for y, x in product(range(i, i + a), range(j, j + b)):
                    grid[y][x] = False
            
            if i + b <= H and j + a <= W and all(not grid[y][x] for y, x in product(range(i, i + b), range(j, j + a))):
                for y, x in product(range(i, i + b), range(j, j + a)):
                    grid[y][x] = True
                if dfs(grid, tiles[:k] + tiles[k+1:], i, j + a):
                    return True
                for y, x in product(range(i, i + b), range(j, j + a)):
                    grid[y][x] = False
        
        return False

    grid = [[False] * W for _ in range(H)]
    return dfs(grid, tiles, 0, 0)

N, H, W = map(int, input().split())
tiles = [tuple(map(int, input().split())) for _ in range(N)]

print("Yes" if can_cover_grid(H, W, tiles) else "No")