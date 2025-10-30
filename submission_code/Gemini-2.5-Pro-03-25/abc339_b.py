# YOUR CODE HERE
import sys

def solve():
    """
    Solves the grid painting and movement problem.
    Reads input H, W, N.
    Simulates N steps of Takahashi's movement and painting on a toroidal HxW grid.
    Prints the final state of the grid.
    """
    H, W, N = map(int, sys.stdin.readline().split())

    # Initialize the grid with all white cells ('.')
    # grid[r][c] represents the cell at row r, column c (0-based indexing)
    grid = [['.' for _ in range(W)] for _ in range(H)]

    # Initial position (0-based indexing: row 0, column 0 corresponds to problem's (1, 1))
    r, c = 0, 0
    
    # Initial direction: Up
    # We use integer representation for direction:
    # 0: Up   (change in row: -1, change in col:  0)
    # 1: Right (change in row:  0, change in col:  1)
    # 2: Down  (change in row:  1, change in col:  0)
    # 3: Left  (change in row:  0, change in col: -1)
    direction = 0

    # Define the changes in row and column for each direction (delta arrays)
    # dr[direction] gives the change in row for moving in that direction
    # dc[direction] gives the change in column for moving in that direction
    dr = [-1, 0, 1, 0]  # Changes in row for Up, Right, Down, Left respectively
    dc = [0, 1, 0, -1]  # Changes in column for Up, Right, Down, Left respectively

    # Simulate N operations
    for _ in range(N):
        # Check the color of the current cell (r, c)
        if grid[r][c] == '.':
            # --- Action for a WHITE cell ---
            # 1. Repaint it black ('#')
            grid[r][c] = '#'
            # 2. Rotate 90 degrees clockwise (e.g., Up -> Right, Right -> Down, ...)
            # Modulo 4 ensures the direction stays within 0, 1, 2, 3
            direction = (direction + 1) % 4
            # 3. Move forward one cell in the *new* direction
            # The movement must wrap around due to the toroidal grid.
            # Adding H or W before the modulo handles potential negative results correctly
            # and ensures the result is within [0, H-1] or [0, W-1].
            r = (r + dr[direction] + H) % H
            c = (c + dc[direction] + W) % W
        else: # grid[r][c] == '#'
            # --- Action for a BLACK cell ---
            # 1. Repaint it white ('.')
            grid[r][c] = '.'
            # 2. Rotate 90 degrees counterclockwise (e.g., Up -> Left, Left -> Down, ...)
            # Add 4 before taking modulo to handle the case when direction is 0 (0 - 1 = -1)
            direction = (direction - 1 + 4) % 4
            # 3. Move forward one cell in the *new* direction
            # Apply toroidal boundary conditions using modulo arithmetic
            r = (r + dr[direction] + H) % H
            c = (c + dc[direction] + W) % W

    # Print the final state of the grid
    for i in range(H):
        # Join the characters in the i-th row list to form the output string for that row
        print("".join(grid[i]))

# Call the main function to execute the solution
solve()