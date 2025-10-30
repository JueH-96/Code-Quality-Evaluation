# YOUR CODE HERE
def check_tak_code(grid, start_row, start_col):
    # Check if 9x9 region starting at (start_row, start_col) is a valid TaK Code
    
    # Check top-left 3x3 region (all must be black)
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check bottom-right 3x3 region (all must be black)
    for i in range(6, 9):
        for j in range(6, 9):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check cells adjacent to top-left 3x3 region
    # Row 3, columns 0-3
    for j in range(4):
        if grid[start_row + 3][start_col + j] != '.':
            return False
    
    # Column 3, rows 0-3
    for i in range(4):
        if grid[start_row + i][start_col + 3] != '.':
            return False
    
    # Check cells adjacent to bottom-right 3x3 region
    # Row 5, columns 5-8
    for j in range(5, 9):
        if grid[start_row + 5][start_col + j] != '.':
            return False
    
    # Column 5, rows 5-8
    for i in range(5, 9):
        if grid[start_row + i][start_col + 5] != '.':
            return False
    
    return True

# Read input
N, M = map(int, input().split())
grid = []
for _ in range(N):
    grid.append(input().strip())

# Find all valid TaK Code regions
results = []
for i in range(N - 8):  # Can start from row 0 to N-9
    for j in range(M - 8):  # Can start from col 0 to M-9
        if check_tak_code(grid, i, j):
            results.append((i + 1, j + 1))  # Convert to 1-indexed

# Output results
for row, col in results:
    print(row, col)