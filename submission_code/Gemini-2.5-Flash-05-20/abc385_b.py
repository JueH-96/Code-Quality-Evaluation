# YOUR CODE HERE
H, W, X, Y = map(int, input().split())

grid = []
for _ in range(H):
    grid.append(input())

T = input()

# Convert initial position to 0-indexed
current_r = X - 1
current_c = Y - 1

# Use a set to store distinct house coordinates (0-indexed)
visited_houses = set()

# Define movement deltas for U, D, L, R
move_deltas = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

# Simulate Santa's movements
for move_char in T:
    dr, dc = move_deltas[move_char]
    
    next_r = current_r + dr
    next_c = current_c + dc
    
    # Check if the next cell is passable.
    # Constraints: S_i,1, S_i,W, S_1,j, S_H,j are '#' for all i,j.
    # This means the grid has a solid '#' border, so we don't need explicit
    # bounds checks (0 <= next_r < H, etc.) as Santa will always hit a '#'
    # wall before going out of the allocated grid memory.
    
    if grid[next_r][next_c] != '#':
        # Move is successful
        current_r = next_r
        current_c = next_c
        
        # Check if the new cell contains a house ('@')
        if grid[current_r][current_c] == '@':
            # Add the 0-indexed coordinates of the house to the set
            visited_houses.add((current_r, current_c))
    # Else (if grid[next_r][next_c] is '#'), Santa stays in his current position
    # and nothing changes for this move.

# Convert final position back to 1-indexed for output
final_X = current_r + 1
final_Y = current_c + 1
distinct_house_count = len(visited_houses)

# Print the results
print(final_X, final_Y, distinct_house_count)