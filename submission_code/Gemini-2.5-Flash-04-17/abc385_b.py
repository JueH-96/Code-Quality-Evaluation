import sys

# Read input using sys.stdin.readline for potentially faster I/O
H, W, X, Y = map(int, sys.stdin.readline().split())

# Convert 1-indexed input coordinates (X, Y) to 0-indexed (start_x, start_y)
start_x, start_y = X - 1, Y - 1

# Read the grid S. S will be a list of strings.
S = [sys.stdin.readline().strip() for _ in range(H)]

# Read the movement string T
T = sys.stdin.readline().strip()

# Initialize Santa's current position in 0-indexed coordinates
current_x, current_y = start_x, start_y

# Use a set to store the coordinates of distinct houses visited
# A tuple (row, col) is used as the key in the set
visited_houses = set()

# The problem guarantees S[X,Y] = '.', so the starting cell
# does not contain a house. We only count houses upon entering
# a cell *during* the movement actions.

# Iterate through each movement instruction in T
for move in T:
    # Determine the target cell coordinates based on the current position
    # and the movement instruction. Initialize target to current position,
    # assuming Santa stays put unless a valid move is made.
    target_x, target_y = current_x, current_y

    if move == 'U':
        target_x -= 1
    elif move == 'D':
        target_x += 1
    elif move == 'L':
        target_y -= 1
    elif move == 'R':
        target_y += 1

    # Check if the target cell is within the grid boundaries
    # and if it is passable (not '#').
    # The grid has H rows (0 to H-1) and W columns (0 to W-1).
    is_target_valid = False
    if 0 <= target_x < H and 0 <= target_y < W:
        if S[target_x][target_y] != '#':
            is_target_valid = True

    # If the target cell is valid and passable:
    if is_target_valid:
        # Update Santa's current position to the target cell
        current_x, current_y = target_x, target_y

        # Check if the cell Santa just moved into contains a house ('@')
        if S[current_x][current_y] == '@':
            # If it's a house, add its coordinates to the set of visited houses.
            # The set automatically handles uniqueness.
            visited_houses.add((current_x, current_y))

    # If the target cell is not valid (out of bounds or impassable),
    # Santa stays in his current cell, so current_x and current_y
    # are not updated.

# After processing all instructions in T, Santa is at (current_x, current_y).
# Convert the final position back to 1-indexed coordinates for output.
final_X_1_indexed = current_x + 1
final_Y_1_indexed = current_y + 1

# The number of distinct houses visited is the size of the visited_houses set.
house_count = len(visited_houses)

# Print the final 1-indexed coordinates and the house count, separated by spaces.
print(final_X_1_indexed, final_Y_1_indexed, house_count)