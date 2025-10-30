# YOUR CODE HERE
import sys

# Function to generate the carpet grid iteratively
def generate_carpet(N):
    """
    Generates a level-N carpet grid iteratively.

    Args:
        N: A non-negative integer representing the level of the carpet.

    Returns:
        A list of lists of strings (characters), representing the grid.
    """
    if N == 0:
        # Base case: level 0 is a 1x1 grid with '#'
        return [['#']]

    # Start with the level 0 carpet grid. This variable will hold the grid
    # for the current level we are building upon in the loop.
    # We start with level 0 and build up to level N.
    current_carpet_grid = [['#']]

    # Build carpets iteratively from level 1 up to N
    for k in range(1, N + 1):
        # Size of the smaller carpet from the previous level (k-1)
        sub_size = pow(3, k - 1)
        # Size of the current carpet (level k)
        current_size = pow(3, k)

        # Create a new grid for the current level k.
        # Initialize with placeholders. This will be filled by copying subgrids
        # or filling with '.'.
        next_carpet_grid = [['' for _ in range(current_size)] for _ in range(current_size)]

        # Iterate through the 3x3 block positions that make up the current grid.
        # sr = subgrid row index (0, 1, or 2)
        # sc = subgrid col index (0, 1, or 2)
        for sr in range(3):
            for sc in range(3):
                # Calculate the starting row and column in the current grid
                # for the top-left corner of the block (sr, sc).
                start_row = sr * sub_size
                start_col = sc * sub_size

                # Check if this is the central block (row 1, column 1) in the 3x3 layout
                if sr == 1 and sc == 1:
                    # The central block of a level-k carpet (for k>0) is entirely white ('.')
                    for r in range(sub_size):
                        for c in range(sub_size):
                            next_carpet_grid[start_row + r][start_col + c] = '.'
                else:
                    # The other eight blocks are level-(k-1) carpets.
                    # Copy the content from the current_carpet_grid (which holds the level k-1 carpet).
                    # The coordinates (r, c) in the previous level's grid correspond
                    # to the coordinates (start_row + r, start_col + c) in the new grid.
                    for r in range(sub_size):
                        for c in range(sub_size):
                            next_carpet_grid[start_row + r][start_col + c] = current_carpet_grid[r][c]

        # After filling all 9 blocks, the next_carpet_grid is the complete carpet for level k.
        # Update the current_carpet_grid variable to be used in the next iteration (for level k+1).
        current_carpet_grid = next_carpet_grid

    # After the loop finishes (when k reaches N), current_carpet_grid holds the level-N carpet.
    return current_carpet_grid

# Read the input integer N from standard input
N = int(sys.stdin.readline())

# Generate the level-N carpet grid using the iterative approach
carpet_grid = generate_carpet(N)

# Print the generated grid row by row to standard output
for row in carpet_grid:
    # Join the list of characters in the current row into a single string
    print("".join(row))