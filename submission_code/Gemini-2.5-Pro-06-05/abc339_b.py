# YOUR CODE HERE
def solve():
    """
    Solves the Langton's Ant problem on a toroidal grid.
    """
    try:
        # Read grid dimensions and number of operations
        H, W, N = map(int, input().split())
    except (IOError, ValueError):
        # Handle cases with no input
        return

    # Initialize the grid. 0 represents a white cell, 1 represents a black cell.
    grid = [[0 for _ in range(W)] for _ in range(H)]

    # Takahashi's initial state
    # r, c: 0-indexed row and column. Start at (1, 1) -> (0, 0).
    r, c = 0, 0
    # direction: 0:Up, 1:Right, 2:Down, 3:Left. Starts facing Up.
    direction = 0

    # Direction vectors (dr, dc) for Up, Right, Down, Left.
    # The order corresponds to a 90-degree clockwise rotation.
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]

    # Perform the N operations
    for _ in range(N):
        # Check the color of the current cell
        if grid[r][c] == 0:  # If white
            grid[r][c] = 1  # Repaint black
            direction = (direction + 1) % 4  # Rotate 90 degrees clockwise
        else:  # If black
            grid[r][c] = 0  # Repaint white
            # Rotate 90 degrees counter-clockwise
            # Adding 4 before modulo handles negative results gracefully.
            direction = (direction - 1 + 4) % 4
        
        # Move forward one cell in the new direction.
        # The modulo operator handles the toroidal wrap-around.
        r = (r + dr[direction]) % H
        c = (c + dc[direction]) % W

    # Print the final state of the grid
    for row in grid:
        # Convert the row of 0s and 1s to a string of '.' and '#'
        print("".join(['#' if cell == 1 else '.' for cell in row]))

# Run the solution
solve()