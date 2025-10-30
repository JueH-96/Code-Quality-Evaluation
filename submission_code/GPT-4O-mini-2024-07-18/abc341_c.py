def possible_positions(H, W, N, T, grid):
    # Directions mapping
    directions = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0)
    }
    
    # Set to store all possible starting positions
    possible_starts = set()
    
    # Iterate through the grid to find all land cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                possible_starts.add((i, j))
    
    # Function to check if a position is valid
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'

    # Check each possible starting position
    count = 0
    for start in possible_starts:
        x, y = start
        
        # Simulate the moves
        valid = True
        for move in T:
            dx, dy = directions[move]
            x += dx
            y += dy
            
            if not is_valid(x, y):
                valid = False
                break
        
        if valid:
            count += 1
    
    return count

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W, N = map(int, data[0].split())
T = data[1]
grid = data[2:H+2]

# Get the result
result = possible_positions(H, W, N, T, grid)

# Print the result
print(result)