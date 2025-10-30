from collections import deque

def collapse_row(row):
    q = deque()
    i = 0
    while i < len(row):
        j = i + 1
        while j < len(row) and row[i] == row[j]:
            j += 1
        if j - i >= 2:
            q.extend([0] * (j - i))
        else:
            q.append(row[i])
        i = j
    return q

def collapse_column(grid, col):
    q = deque()
    i = 0
    while i < len(grid):
        j = i + 1
        while j < len(grid) and grid[j][col] == grid[i][col]:
            j += 1
        if j - i >= 2:
            for k in range(i, j):
                grid[k][col] = 0
        else:
            q.append(grid[i][col])
        i = j
    return q

def collapse(grid):
    new_grid = []
    for row in grid:
        new_grid.append(collapse_row(row))
    for col in range(len(grid[0])):
        collapse_column(new_grid, col)
    return new_grid

def main():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(input()))
    while True:
        new_grid = collapse(grid)
        if all(len(row) == len(grid[0]) for row in new_grid):
            break
        grid = []
        for row in new_grid:
            grid.append(list(row))
    ans = 0
    for row in grid:
        ans += sum(1 for x in row if x != 0)
    print(ans)

main()