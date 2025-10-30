# Read input
H, W, N = map(int, input().split())
T = input().strip()
grid = [input().strip() for _ in range(H)]

possible_positions = set()

# Iterate over each cell in the grid
for i in range(1, H + 1):
    for j in range(1, W + 1):
        # Check if the current cell is land
        if grid[i-1][j-1] == '.':
            current_i, current_j = i, j
            valid = True
            # Simulate each move
            for move in T:
                if move == 'L':
                    new_i, new_j = current_i, current_j - 1
                elif move == 'R':
                    new_i, new_j = current_i, current_j + 1
                elif move == 'U':
                    new_i, new_j = current_i - 1, current_j
                elif move == 'D':
                    new_i, new_j = current_i + 1, current_j
                else:
                    valid = False
                    break
                # Check if new position is within grid bounds
                if new_i < 1 or new_i > H or new_j < 1 or new_j > W:
                    valid = False
                    break
                # Check if new position is land
                if grid[new_i - 1][new_j - 1] == '#':
                    valid = False
                    break
                # Update current position
                current_i, current_j = new_i, new_j
            if valid:
                possible_positions.add((current_i, current_j))

# Output the number of possible positions
print(len(possible_positions))