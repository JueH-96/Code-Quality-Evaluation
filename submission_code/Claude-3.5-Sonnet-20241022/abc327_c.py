def check_row_col(grid):
    # Check each row and column
    for i in range(9):
        row = set(grid[i])
        col = set(grid[j][i] for j in range(9))
        if len(row) != 9 or len(col) != 9:
            return False
    return True

def check_3x3_grids(grid):
    # Check each 3x3 subgrid
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = set()
            for i in range(3):
                for j in range(3):
                    box.add(grid[box_row + i][box_col + j])
            if len(box) != 9:
                return False
    return True

# Read input
grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Check all conditions
if check_row_col(grid) and check_3x3_grids(grid):
    print("Yes")
else:
    print("No")