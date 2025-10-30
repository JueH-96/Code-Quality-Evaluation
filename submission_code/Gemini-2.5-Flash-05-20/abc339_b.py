# YOUR CODE HERE
import sys

def solve():
    H, W, N = map(int, sys.stdin.readline().split())

    # Initialize the grid with all white cells ('.')
    grid = [['.' for _ in range(W)] for _ in range(H)]

    # Robot's initial state
    curr_row = 0
    curr_col = 0
    
    # Directions: 0=Up, 1=Right, 2=Down, 3=Left
    # DR: change in row for each direction
    # DC: change in column for each direction
    DR = [-1, 0, 1, 0] # Corresponds to Up, Right, Down, Left
    DC = [0, 1, 0, -1] # Corresponds to Up, Right, Down, Left
    
    curr_direction = 0 # Initially facing Up (direction index 0)

    # Perform N operations
    for _ in range(N):
        # Get current cell color
        current_cell_color = grid[curr_row][curr_col]

        if current_cell_color == '.': # If current cell is white
            # Repaint black
            grid[curr_row][curr_col] = '#'
            # Rotate 90 degrees clockwise
            curr_direction = (curr_direction + 1) % 4
        else: # If current cell is black ('#')
            # Repaint white
            grid[curr_row][curr_col] = '.'
            # Rotate 90 degrees counterclockwise
            # Adding 4 before modulo handles the case where curr_direction is 0 (Up)
            # (0 - 1 + 4) % 4 = 3, which is Left
            curr_direction = (curr_direction - 1 + 4) % 4

        # Move forward one cell in the new direction
        # The + H and + W before modulo handle wrapping around for negative results
        # (e.g., moving up from row 0 wraps to H-1)
        curr_row = (curr_row + DR[curr_direction] + H) % H
        curr_col = (curr_col + DC[curr_direction] + W) % W

    # Print the final grid
    for r in range(H):
        sys.stdout.write("".join(grid[r]) + "
")

# Call the solve function to run the program
solve()