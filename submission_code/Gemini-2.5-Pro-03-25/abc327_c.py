# YOUR CODE HERE
import sys

def solve():
    """
    Reads a 9x9 grid from standard input and checks if it satisfies the conditions
    of a Sudoku puzzle solution.
    Prints "Yes" if it satisfies all conditions, otherwise prints "No".
    """
    grid = []
    # Read the 9 rows of the grid from standard input
    for _ in range(9):
        try:
            # Read a line, strip potential whitespace, split by space, and convert to integers
            line = sys.stdin.readline()
            if not line:
                 # Handle unexpected end of input
                 # According to typical competitive programming constraints,
                 # this case might not occur if input is guaranteed valid.
                 # If it occurs, it implies invalid input format.
                 print("No") # Or handle as an error
                 return

            row = list(map(int, line.split()))

            # Check if the row has exactly 9 numbers
            if len(row) != 9:
                 # Invalid row length means the grid format is wrong.
                 print("No") # Or handle as an error
                 return

            grid.append(row)
        except ValueError:
            # Handle cases where conversion to int fails (non-integer input)
            print("No") # Or handle as an error
            return
        except Exception:
             # Catch other potential unexpected errors during input reading
             print("No") # Or handle as an error
             return

    # Define the target set of numbers that must appear in each row, column, and block
    target_set = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    # 1. Check rows
    # Iterate through each row of the grid
    for r in range(9):
        # Convert the list of numbers in the current row to a set.
        # A set automatically handles duplicates and allows easy comparison.
        row_set = set(grid[r])
        # Check if the set of numbers in the row is exactly the target set {1, ..., 9}.
        # This verifies two things simultaneously:
        # - All numbers are between 1 and 9 (inclusive).
        # - Each number from 1 to 9 appears exactly once (no duplicates, none missing).
        if row_set != target_set:
            # If the condition fails for any row, the grid is invalid.
            print("No")
            return # Exit the function early

    # 2. Check columns
    # Iterate through each column index
    for c in range(9):
        col_set = set()
        # Iterate through each row to access the element at the current column index
        for r in range(9):
            col_set.add(grid[r][c]) # Add the element to the column set
        # Check if the set of numbers in the column matches the target set.
        if col_set != target_set:
            # If the condition fails for any column, the grid is invalid.
            print("No")
            return # Exit the function early

    # 3. Check 3x3 blocks
    # Iterate through the starting row index of each 3x3 block (0, 3, 6)
    for block_row_start in range(0, 9, 3):
        # Iterate through the starting column index of each 3x3 block (0, 3, 6)
        for block_col_start in range(0, 9, 3):
            block_set = set()
            # Iterate through the 3 rows within the current block
            for r in range(block_row_start, block_row_start + 3):
                # Iterate through the 3 columns within the current block
                for c in range(block_col_start, block_col_start + 3):
                    # Add the element in the current cell to the block set
                    block_set.add(grid[r][c])
            # Check if the set of numbers in the block matches the target set.
            if block_set != target_set:
                # If the condition fails for any block, the grid is invalid.
                print("No")
                return # Exit the function early

    # If all three checks (rows, columns, blocks) have passed without returning "No",
    # it means the grid satisfies all conditions.
    print("Yes")

# Call the main function to start the process
solve()