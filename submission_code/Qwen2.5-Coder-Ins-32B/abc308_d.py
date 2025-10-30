# YOUR CODE HERE
def is_valid(x, y, H, W):
    return 0 <= x < H and 0 <= y < W

def dfs(x, y, step, grid, H, W):
    if step == 8:
        step = 3
    target = 'snuke'[step]
    if grid[x][y] != target:
        return False
    if x == H - 1 and y == W - 1:
        return True
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny, H, W) and (nx, ny) not in visited:
            visited.add((nx, ny))
            if dfs(nx, ny, (step + 1) % 5, grid, H, W):
                return True
            visited.remove((nx, ny))
    return False

import sys
input = sys.stdin.read().split()
H, W = map(int, input[:2])
grid = input[2:]

visited = set()
visited.add((0, 0))
if dfs(0, 0, 0, grid, H, W):
    print("Yes")
else:
    print("No")