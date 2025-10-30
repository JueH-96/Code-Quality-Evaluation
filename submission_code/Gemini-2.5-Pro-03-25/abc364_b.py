# YOUR CODE HERE
import sys

def solve():
    # Read grid dimensions H (height/rows) and W (width/columns)
    H, W = map(int, sys.stdin.readline().split())

    # Read starting position (1-based indexing) S_i (row) and S_j (column)
    S_i, S_j = map(int, sys.stdin.readline().split())

    # Read the grid configuration. Each line represents a row.
    # grid[r][c] stores the character ('#' or '.') at row r+1, column c+1
    # We use 0-based indexing internally, so grid[r] is the (r+1)-th row string.
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Read the sequence of actions X
    X = sys.stdin.readline().strip()

    # Convert the starting position from 1-based to 0-based indexing
    # cur_r will range from 0 to H-1
    # cur_c will range from 0 to W-1
    cur_r = S_i - 1
    cur_c = S_j - 1

    # Simulate Takahashi's actions one by one according to the string X
    for action in X:
        # Calculate the potential next position based on the current action
        # Initialize potential next position with the current position
        next_r, next_c = cur_r, cur_c

        # Update potential next position based on the action character
        if action == 'L':
            next_c -= 1 # Attempt to move left
        elif action == 'R':
            next_c += 1 # Attempt to move right
        elif action == 'U':
            next_r -= 1 # Attempt to move up
        elif action == 'D':
            next_r += 1 # Attempt to move down

        # Check if the potential next position is valid for a move:
        # Condition 1: The potential next position must be within the grid boundaries.
        # Row index (next_r) must be between 0 and H-1 (inclusive).
        # Column index (next_c) must be between 0 and W-1 (inclusive).
        is_within_bounds = (0 <= next_r < H) and (0 <= next_c < W)

        # Condition 2: If the potential position is within bounds,
        # the cell at that position must be empty (represented by '.').
        if is_within_bounds:
            # Access the character at the potential next cell in the grid
            is_empty = (grid[next_r][next_c] == '.')

            # If both conditions are met (within bounds AND empty cell),
            # Takahashi moves to the new position. Update the current position.
            if is_empty:
                cur_r, cur_c = next_r, next_c
        # Else (if the potential position is out of bounds or the cell contains '#'):
            # Takahashi stays in the current cell. We don't update cur_r, cur_c.

    # After iterating through all actions in X, (cur_r, cur_c) holds the final 0-based position.
    # Convert the final position back to 1-based indexing for the output.
    final_r = cur_r + 1
    final_c = cur_c + 1

    # Print the final 1-based coordinates (row and column), separated by a space.
    print(final_r, final_c)

# Ensure the solve function is called when the script is executed
if __name__ == '__main__':
    solve()