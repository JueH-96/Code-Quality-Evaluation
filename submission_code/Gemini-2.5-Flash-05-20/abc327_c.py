import sys

# Function to read the 9x9 grid from standard input
def read_grid():
    grid = []
    for _ in range(9):
        # Read a line, split it by space, convert each part to an integer,
        # and append the resulting list of integers as a row to the grid.
        row = list(map(int, sys.stdin.readline().split()))
        grid.append(row)
    return grid

# Helper function to check if a list of 9 numbers contains each integer
# from 1 to 9 exactly once.
def is_valid_set(numbers):
    # A standard Sudoku segment (row, column, or 3x3 block) must contain
    # all digits from 1 to 9 exactly once.
    # By converting the list to a set, we automatically get unique elements.
    # If the set of numbers is exactly equal to the set of numbers from 1 to 9,
    # it means all numbers are present and unique.
    expected_set = set(range(1, 10))
    return set(numbers) == expected_set

# Condition 1: Check if all rows are valid
def check_rows(grid):
    for r in range(9):
        # Pass each row of the grid to the helper function
        if not is_valid_set(grid[r]):
            return False  # If any row is invalid, the condition fails
    return True  # All rows are valid

# Condition 2: Check if all columns are valid
def check_columns(grid):
    for c in range(9):
        # Extract elements for the current column
        column = [grid[r][c] for r in range(9)]
        # Pass the extracted column to the helper function
        if not is_valid_set(column):
            return False  # If any column is invalid, the condition fails
    return True  # All columns are valid

# Condition 3: Check if all nine 3x3 subgrids are valid
def check_subgrids(grid):
    # Iterate through the 3x3 blocks. There are 3 blocks horizontally and 3 vertically.
    # block_row and block_col will range from 0 to 2, representing the index of the 3x3 block.
    for block_row in range(3):
        for block_col in range(3):
            subgrid_elements = []
            # Calculate the starting row and column index for the current 3x3 block
            start_row = block_row * 3
            start_col = block_col * 3

            # Iterate through the cells within the current 3x3 block
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    subgrid_elements.append(grid[r][c])
            
            # Pass the collected elements of the 3x3 subgrid to the helper function
            if not is_valid_set(subgrid_elements):
                return False  # If any 3x3 subgrid is invalid, the condition fails
    return True  # All 3x3 subgrids are valid

# Main function to solve the problem
def solve():
    grid = read_grid()

    # Check all three conditions. If all are true, the grid is valid.
    # Otherwise, it's invalid.
    if check_rows(grid) and check_columns(grid) and check_subgrids(grid):
        print("Yes")
    else:
        print("No")

# Ensure the solve function is called when the script is executed
if __name__ == "__main__":
    solve()