# YOUR CODE HERE
def is_tak_code(grid, i, j):
    # Check top-left 3x3
    if not all(grid[r][c] == '#' for r in range(i, i+3) for c in range(j, j+3)):
        return False
    
    # Check bottom-right 3x3
    if not all(grid[r][c] == '#' for r in range(i+6, i+9) for c in range(j+6, j+9)):
        return False
    
    # Check white cells adjacent to top-left 3x3
    for r in range(i, i+4):
        if grid[r][j+3] != '.':
            return False
    for c in range(j, j+4):
        if grid[i+3][c] != '.':
            return False
    if grid[i+3][j+3] != '.':
        return False
    
    # Check white cells adjacent to bottom-right 3x3
    for r in range(i+5, i+9):
        if grid[r][j+5] != '.':
            return False
    for c in range(j+5, j+9):
        if grid[i+5][c] != '.':
            return False
    if grid[i+5][j+5] != '.':
        return False
    
    return True

N, M = map(int, input().split())
grid = [input() for _ in range(N)]

tak_codes = []
for i in range(N - 8):
    for j in range(M - 8):
        if is_tak_code(grid, i, j):
            tak_codes.append((i+1, j+1))

for i, j in tak_codes:
    print(i, j)