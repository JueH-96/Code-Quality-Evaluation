import sys

def read_input():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H, W = int(data[idx]), int(data[idx + 1])
    idx += 2
    grid = data[idx:idx + H]
    return H, W, grid

def remove_cookies(grid):
    H, W = len(grid), len(grid[0])
    while True:
        marked = set()
        # Mark rows
        for i in range(H):
            if len(set(grid[i])) == 1 and len(set(grid[i])) > 1:
                marked.update([(i, j) for j in range(W)])
        # Mark columns
        for j in range(W):
            column = ''.join([grid[i][j] for i in range(H)])
            if len(set(column)) == 1 and len(set(column)) > 1:
                marked.update([(i, j) for i in range(H)])
        if not marked:
            break
        # Remove marked cookies
        new_grid = []
        for i in range(H):
            row = ''.join([grid[i][j] for j in range(W) if (i, j) not in marked])
            if row:
                new_grid.append(row)
        grid = new_grid
        H = len(grid)
    return sum(len(row) for row in grid)

H, W, grid = read_input()
result = remove_cookies(grid)
print(result)