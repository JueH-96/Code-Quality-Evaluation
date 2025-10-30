# Read the 9x9 grid
grid = []
for i in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Check if a list has numbers 1-9 exactly once
def is_valid_sequence(seq):
    return sorted(seq) == list(range(1, 10))

# Check condition 1: each row has 1-9 exactly once
def check_rows(grid):
    for row in grid:
        if not is_valid_sequence(row):
            return False
    return True

# Check condition 2: each column has 1-9 exactly once
def check_columns(grid):
    for j in range(9):
        column = [grid[i][j] for i in range(9)]
        if not is_valid_sequence(column):
            return False
    return True

# Check condition 3: each 3x3 subgrid has 1-9 exactly once
def check_subgrids(grid):
    for block_row in range(3):
        for block_col in range(3):
            subgrid = []
            for i in range(3):
                for j in range(3):
                    subgrid.append(grid[block_row * 3 + i][block_col * 3 + j])
            if not is_valid_sequence(subgrid):
                return False
    return True

# Check all conditions
if check_rows(grid) and check_columns(grid) and check_subgrids(grid):
    print("Yes")
else:
    print("No")