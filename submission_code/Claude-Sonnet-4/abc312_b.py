def solve():
    N, M = map(int, input().split())
    grid = []
    for _ in range(N):
        grid.append(input().strip())
    
    results = []
    
    # Check all possible 9x9 regions
    for i in range(N - 8):  # top-left row can be from 0 to N-9
        for j in range(M - 8):  # top-left col can be from 0 to M-9
            if is_tak_code(grid, i, j):
                results.append((i + 1, j + 1))  # Convert to 1-indexed
    
    for r in results:
        print(r[0], r[1])

def is_tak_code(grid, start_row, start_col):
    # Check top-left 3x3 region - must be all black
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check bottom-right 3x3 region - must be all black
    for i in range(6, 9):
        for j in range(6, 9):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check adjacent cells to top-left 3x3 - must be white
    # Right edge of top-left 3x3
    for i in range(3):
        if grid[start_row + i][start_col + 3] != '.':
            return False
    
    # Bottom edge of top-left 3x3
    for j in range(4):  # includes the corner cell (3,3)
        if grid[start_row + 3][start_col + j] != '.':
            return False
    
    # Check adjacent cells to bottom-right 3x3 - must be white
    # Left edge of bottom-right 3x3
    for i in range(6, 9):
        if grid[start_row + i][start_col + 5] != '.':
            return False
    
    # Top edge of bottom-right 3x3
    for j in range(5, 9):  # includes the corner cell (5,5)
        if grid[start_row + 5][start_col + j] != '.':
            return False
    
    return True

solve()