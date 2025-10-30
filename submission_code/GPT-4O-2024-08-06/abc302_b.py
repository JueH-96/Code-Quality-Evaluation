# YOUR CODE HERE
def find_snuke(grid, H, W):
    # Directions: (dx, dy) pairs
    directions = [
        (0, 1),   # right
        (1, 0),   # down
        (1, 1),   # down-right
        (1, -1),  # down-left
        (0, -1),  # left
        (-1, 0),  # up
        (-1, -1), # up-left
        (-1, 1)   # up-right
    ]
    
    # The word we are looking for
    word = "snuke"
    
    # Iterate over each cell in the grid
    for i in range(H):
        for j in range(W):
            # Check if the current cell is 's'
            if grid[i][j] == 's':
                # Try each direction
                for dx, dy in directions:
                    # Check if we can find the word "snuke" in this direction
                    found = True
                    positions = []
                    for k in range(5):
                        ni, nj = i + k * dx, j + k * dy
                        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == word[k]:
                            positions.append((ni + 1, nj + 1))  # Convert to 1-based index
                        else:
                            found = False
                            break
                    if found:
                        return positions

import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:H+2]

result = find_snuke(grid, H, W)
for r, c in result:
    print(r, c)