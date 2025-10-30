def is_tak_code(grid, start_row, start_col):
    # Check if 9x9 region starting at (start_row, start_col) is a TaK Code
    
    # Check top-left 3x3 is all black
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check bottom-right 3x3 is all black
    for i in range(6, 9):
        for j in range(6, 9):
            if grid[start_row + i][start_col + j] != '#':
                return False
    
    # Check all 14 adjacent cells are white
    adjacent_cells = [
        (0,3), (1,3), (2,3), (3,0), (3,1), (3,2), (3,3),
        (5,5), (5,6), (5,7), (5,8), (6,5), (7,5), (8,5)
    ]
    for i, j in adjacent_cells:
        if grid[start_row + i][start_col + j] != '.':
            return False
    
    return True

# Read input
n, m = map(int, input().split())
grid = []
for _ in range(n):
    grid.append(input())

# Find all valid TaK Codes
results = []
for i in range(n - 8):  # 9x9 region starting at row i, so i goes from 0 to n-9
    for j in range(m - 8):  # 9x9 region starting at col j, so j goes from 0 to m-9
        if is_tak_code(grid, i, j):
            results.append((i + 1, j + 1))  # Convert to 1-based indexing

# Output results
for i, j in results:
    print(i, j)