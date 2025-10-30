# YOUR CODE HERE
def is_tak_code(grid, i, j):
    n = len(grid)
    m = len(grid[0])
    if i + 8 >= n or j + 8 >= m:
        return False
    
    # Check top-left 3x3 region
    for x in range(3):
        for y in range(3):
            if grid[i+x][j+y] != '#':
                return False
    
    # Check adjacent cells
    for x in range(9):
        for y in range(9):
            if (x == 0 or y == 0 or x == 8 or y == 8) and grid[i+x][j+y] == '#':
                return False
            if (x == 1 or y == 1 or x == 7 or y == 7) and grid[i+x][j+y] == '.':
                return False
            if (x == 2 or y == 2 or x == 6 or y == 6) and grid[i+x][j+y] != '#':
                return False
            if (x == 3 or y == 3 or x == 5 or y == 5) and grid[i+x][j+y] != '.':
                return False
            if (x == 4 and y == 4) and grid[i+x][j+y] == '.':
                return False
    
    return True

n, m = map(int, input().split())
grid = [input() for _ in range(n)]

for i in range(n - 8):
    for j in range(m - 8):
        if is_tak_code(grid, i, j):
            print(i+1, j+1)