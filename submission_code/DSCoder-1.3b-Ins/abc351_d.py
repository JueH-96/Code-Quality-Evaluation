H, W = map(int, input().split())
grid = [input() for _ in range(H)]

def count_reachable_cells(x, y):
    if x < 0 or y < 0 or x >= H or y >= W or grid[x][y] == '#':
        return 0
    grid[x][y] = '#'
    return 1 + count_reachable_cells(x-1, y) + count_reachable_cells(x+1, y) + count_reachable_cells(x, y-1) + count_reachable_cells(x, y+1)

reachable_cells = [count_reachable_cells(x, y) for x in range(H) for y in range(W) if grid[x][y] == '.' ]
print(max(reachable_cells))