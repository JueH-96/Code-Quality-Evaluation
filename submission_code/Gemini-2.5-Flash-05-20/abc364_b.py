import sys

def solve():
    # Read H and W
    H, W = map(int, sys.stdin.readline().split())

    # Read starting coordinates S_i, S_j
    # Convert to 0-indexed for internal processing
    S_i, S_j = map(int, sys.stdin.readline().split())
    current_row = S_i - 1
    current_col = S_j - 1

    # Read the grid
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())

    # Read the action string X
    X = sys.stdin.readline().strip()

    # Simulate Takahashi's movement
    for move_char in X:
        # Tentative next position
        next_row = current_row
        next_col = current_col

        # Determine tentative next coordinates based on move_char
        if move_char == 'L':
            next_col -= 1
        elif move_char == 'R':
            next_col += 1
        elif move_char == 'U':
            next_row -= 1
        elif move_char == 'D':
            next_row += 1

        # Check if the tentative next position is within grid boundaries
        # and if the cell at that position is empty ('.')
        can_move = False
        if 0 <= next_row < H and 0 <= next_col < W:
            if grid[next_row][next_col] == '.':
                can_move = True
        
        # If movement is valid, update current position
        if can_move:
            current_row = next_row
            current_col = next_col
        # Otherwise, Takahashi stays in the current cell (current_row, current_col remain unchanged)

    # Print the final position, converting back to 1-indexed
    print(f"{current_row + 1} {current_col + 1}")

# Call the solve function to run the program
solve()