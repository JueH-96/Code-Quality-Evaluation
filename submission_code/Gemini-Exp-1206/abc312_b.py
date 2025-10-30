def is_tak_code(grid, r, c):
    for i in range(r, r + 3):
        for j in range(c, c + 3):
            if grid[i][j] == '.':
                return False
    for i in range(r + 6, r + 9):
        for j in range(c + 6, c + 9):
            if grid[i][j] == '.':
                return False
    for i in range(r - 1, r + 4):
        if r -1 <= i < r + 9:
            if c - 1 >= 0 and grid[i][c - 1] == '#':
                return False
            if c + 3 < len(grid[0]) and grid[i][c + 3] == '#':
                return False
    for j in range(c - 1, c + 4):
        if c - 1 <= j < c + 9:
            if r - 1 >= 0 and grid[r - 1][j] == '#':
                return False
            if r + 3 < len(grid) and grid[r + 3][j] == '#':
                return False
    
    for i in range(r + 5, r + 10):
        if r -1 <= i < r + 9:
            if c + 5 >= 0 and grid[i][c + 5] == '#':
                return False
            if c + 9 < len(grid[0]) and grid[i][c + 9] == '#':
                return False
    for j in range(c + 5, c + 10):
        if c - 1 <= j < c + 9:
            if r + 5 >= 0 and grid[r + 5][j] == '#':
                return False
            if r + 9 < len(grid) and grid[r + 9][j] == '#':
                return False

    if r + 3 < len(grid) and c + 3 < len(grid[0]) and grid[r+3][c+3] == '#':
        return False
    if r + 5 >= 0 and c + 5 >= 0 and grid[r+5][c+5] == '#':
        return False

    return True

n, m = map(int, input().split())
grid = [input() for _ in range(n)]
result = []
for i in range(n - 8):
    for j in range(m - 8):
        if is_tak_code(grid, i, j):
            result.append((i + 1, j + 1))
for r, c in result:
    print(r, c)