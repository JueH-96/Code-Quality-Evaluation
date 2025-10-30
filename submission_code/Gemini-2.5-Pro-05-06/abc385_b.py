# YOUR CODE HERE
H, W, X, Y = map(int, input().split())

# Convert to 0-indexed coordinates for internal use
curr_r, curr_c = X - 1, Y - 1

grid = []
for _ in range(H):
    grid.append(input())  # Each row as a string

T = input()  # Movement string

visited_houses = set()

# Define change in row/column for each move direction
move_deltas = {
    'U': (-1, 0),  # Up
    'D': (1, 0),   # Down
    'L': (0, -1),  # Left
    'R': (0, 1)    # Right
}

# Simulate Santa's movements
for move_char in T:
    dr, dc = move_deltas[move_char]
    
    next_r, next_c = curr_r + dr, curr_c + dc
    
    # The problem guarantees a border of '#' cells.
    # Santa starts within this border (on a '.') and cannot move onto '#'.
    # Thus, next_r and next_c will always be valid indices for the grid.
    # We only need to check if the target cell is passable (not '#').
    
    if grid[next_r][next_c] != '#':
        # Move is possible, update Santa's position
        curr_r = next_r
        curr_c = next_c
        
        # If Santa lands on a house, add it to the set of visited houses
        if grid[curr_r][curr_c] == '@':
            visited_houses.add((curr_r, curr_c))
    # Else (target cell is '#'), Santa stays in his current cell. Position does not change.

# Output final position (converted back to 1-indexed) and count of distinct houses
print(f"{curr_r + 1} {curr_c + 1} {len(visited_houses)}")