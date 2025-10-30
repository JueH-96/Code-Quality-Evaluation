# YOUR CODE HERE
import sys

# Read input
H, W = map(int, sys.stdin.readline().split())
grid = []
for _ in range(H):
    grid.append(sys.stdin.readline().strip())

# Target word
target = "snuke"

# Possible directions (dr, dc)
# (dr, dc) represent the change in row and column index for each step
# (0, 1): right
# (0, -1): left
# (1, 0): down
# (-1, 0): up
# (1, 1): down-right (diagonal)
# (1, -1): down-left (diagonal)
# (-1, 1): up-right (diagonal)
# (-1, -1): up-left (diagonal)
directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]

# Iterate through each cell (r, c) as a potential starting point
# r: 0-based row index, from 0 to H-1
# c: 0-based column index, from 0 to W-1
for r in range(H):
    for c in range(W):
        # The first letter of the target word is 's'.
        # If the current cell does not contain 's', it cannot be the start of the sequence.
        if grid[r][c] == target[0]:
            # If the current cell contains 's', iterate through all possible directions
            for dr, dc in directions:
                path = [] # List to store the 1-based coordinates of the potential path cells
                is_valid_path = True # Flag to track if the current path sequence is valid
                
                # Check the 5 cells along the current direction starting from (r, c)
                for k in range(5):
                    curr_r = r + k * dr # Row index of the k-th cell (0-based) in the sequence
                    curr_c = c + k * dc # Column index of the k-th cell (0-based) in the sequence
                    
                    # Check if the k-th cell is within the grid boundaries (0-based)
                    if not (0 <= curr_r < H and 0 <= curr_c < W):
                        is_valid_path = False # Path goes out of bounds
                        break # No need to check further cells in this direction, break the inner loop (over k)
                        
                    # Check if the letter in the k-th cell matches the k-th letter of the target word "snuke"
                    if grid[curr_r][curr_c] != target[k]:
                        is_valid_path = False # Letters don't match the required sequence
                        break # No need to check further cells in this direction, break the inner loop (over k)
                        
                    # If the k-th cell is valid so far (within bounds and correct letter),
                    # store its 1-based coordinates for potential output
                    path.append((curr_r + 1, curr_c + 1))
                
                # After checking all 5 cells (k from 0 to 4), if the path is still valid, we found the unique solution
                if is_valid_path:
                    # Print the 1-based coordinates of the 5 cells, one pair (row, column) per line, separated by a space.
                    for row, col in path:
                        print(row, col)
                    
                    # The problem guarantees a unique solution exists. Once found, we can print it and terminate the program.
                    sys.exit()

# If the program finishes without calling sys.exit(), it means no valid path was found.
# According to the problem constraints ("The given grid has a unique conforming set of cells."),
# this part of the code should not be reached with valid test inputs.
# No explicit action is needed here based on the problem guarantee.