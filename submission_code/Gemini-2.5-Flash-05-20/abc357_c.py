import sys

def solve(k: int) -> list[list[str]]:
    """
    Generates a level-k carpet as a 2D list of characters.
    """
    if k == 0:
        # Base case: Level-0 carpet is a 1x1 black cell
        return [['#']]

    # Recursive step: Get the level-(k-1) carpet
    prev_carpet = solve(k - 1)
    prev_size = len(prev_carpet) # This is 3^(k-1)
    
    # Calculate the size of the current level-k carpet (3^k)
    current_size = 3 * prev_size
    
    # Initialize the current carpet grid with all white cells ('.')
    # This handles the central block being white by default
    current_carpet = [['.' for _ in range(current_size)] for _ in range(current_size)]

    # Iterate through the 3x3 block positions
    for block_row in range(3):
        for block_col in range(3):
            # If it's the central block (row 1, col 1), it remains all white
            if block_row == 1 and block_col == 1:
                continue
            else:
                # For the other eight blocks, copy the level-(k-1) carpet pattern
                # Calculate the starting coordinates for the current sub-block
                start_row = block_row * prev_size
                start_col = block_col * prev_size
                
                # Copy prev_carpet into the current_carpet at the determined position
                for r in range(prev_size):
                    for c in range(prev_size):
                        current_carpet[start_row + r][start_col + c] = prev_carpet[r][c]
                        
    return current_carpet

# Read input N from Standard Input
N = int(sys.stdin.readline().strip())

# Generate the level-N carpet
result_grid = solve(N)

# Print the carpet according to the specified format
for row in result_grid:
    # Join characters in the row list to form a string, then print with a newline
    sys.stdout.write("".join(row) + "
")