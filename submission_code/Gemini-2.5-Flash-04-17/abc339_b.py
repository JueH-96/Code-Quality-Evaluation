# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read input values for grid height (H), width (W), and number of operations (N)
    # Input is read from standard input, space-separated integers.
    H, W, N = map(int, sys.stdin.readline().split())

    # Initialize the grid.
    # It's represented as a 2D list (list of lists) of size H x W.
    # We use boolean values: False represents a white cell ('.') and True represents a black cell ('#').
    # Initially, all cells are painted white (False).
    grid = [[False for _ in range(W)] for _ in range(H)]

    # Initialize the current position of Takahashi.
    # The problem uses 1-based indexing for cells (i, j), starting at (1, 1).
    # We use 0-based indexing for Python lists, so (1, 1) corresponds to grid index (0, 0).
    current_row = 0
    current_col = 0

    # Initialize the current direction Takahashi is facing.
    # We map directions to integers: 0 for Up, 1 for Right, 2 for Down, 3 for Left.
    # Takahashi starts facing upwards.
    direction = 0

    # Define movement vectors for each direction (Up, Right, Down, Left).
    # dr[d] is the change in row index when moving one step in direction d.
    # dc[d] is the change in column index when moving one step in direction d.
    dr = [-1, 0, 1, 0] # Row changes: -1 for Up, 0 for Right/Left, 1 for Down
    dc = [0, 1, 0, -1] # Column changes: 0 for Up/Down, 1 for Right, -1 for Left

    # Simulate the N operations as described in the problem.
    for _ in range(N):
        # Check the color of the current cell at (current_row, current_col).
        is_black = grid[current_row][current_col]

        if not is_black: # If the current cell is white (False)
            # According to the rules:
            # 1. Repaint the current cell black.
            grid[current_row][current_col] = True
            
            # 2. Rotate 90 degrees clockwise.
            # The direction sequence is Up(0) -> Right(1) -> Down(2) -> Left(3) -> Up(0).
            # This is achieved by adding 1 to the current direction and taking modulo 4.
            direction = (direction + 1) % 4
            
            # 3. Move forward one cell in the new direction.
            # Calculate the potential next position based on the new direction.
            next_row = current_row + dr[direction]
            next_col = current_col + dc[direction]
            
            # Apply toroidal wrapping to the new position.
            # The modulo operator (%) handles wrapping around the grid edges.
            # Adding the size (H or W) before modulo handles cases where the potential position is negative (e.g., moving up from row 0).
            current_row = (next_row + H) % H
            current_col = (next_col + W) % W

        else: # If the current cell is black (True)
            # According to the rules:
            # 1. Repaint the current cell white.
            grid[current_row][current_col] = False
            
            # 2. Rotate 90 degrees counterclockwise.
            # The direction sequence is Up(0) -> Left(3) -> Down(2) -> Right(1) -> Up(0).
            # This is achieved by subtracting 1 from the current direction and taking modulo 4.
            # Python's modulo operator handles negative numbers correctly (-1 % 4 is 3).
            direction = (direction - 1) % 4
            
            # 3. Move forward one cell in the new direction.
            # Calculate the potential next position.
            next_row = current_row + dr[direction]
            next_col = current_col + dc[direction]
            
            # Apply toroidal wrapping.
            current_row = (next_row + H) % H
            current_col = (next_col + W) % W

    # After N operations are completed, print the final state of the grid.
    # Print H lines, where each line is a string of length W.
    # Use '.' for white cells (False) and '#' for black cells (True).
    for r in range(H):
        row_str = "" # Initialize an empty string for the current row
        for c in range(W):
            # Append '#' if the cell is black (True), otherwise append '.'
            row_str += '#' if grid[r][c] else '.'
        print(row_str) # Print the completed string for the current row

# Execute the solve function to run the simulation and print the output
solve()