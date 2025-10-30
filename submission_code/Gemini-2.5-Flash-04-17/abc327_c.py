import sys

# Read the 9x9 grid
grid = []
for _ in range(9):
    row = list(map(int, sys.stdin.readline().split()))
    grid.append(row)

# Helper function to check if a list of 9 numbers contains 1-9 exactly once
# Given the problem constraints (1 <= A_ij <= 9),
# this is equivalent to checking if the set of numbers has size 9.
def check_sequence(sequence):
    return len(set(sequence)) == 9

# Check rows
for r in range(9):
    if not check_sequence(grid[r]):
        print("No")
        sys.exit()

# Check columns
for c in range(9):
    column_data = [grid[r][c] for r in range(9)]
    if not check_sequence(column_data):
        print("No")
        sys.exit()

# Check 3x3 blocks
for block_row_start in range(0, 9, 3):
    for block_col_start in range(0, 9, 3):
        block_data = []
        for r in range(block_row_start, block_row_start + 3):
            for c in range(block_col_start, block_col_start + 3):
                block_data.append(grid[r][c])
        if not check_sequence(block_data):
            print("No")
            sys.exit()

# If all checks passed
print("Yes")