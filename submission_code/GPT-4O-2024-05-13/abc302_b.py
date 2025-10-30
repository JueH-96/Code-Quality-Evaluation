# YOUR CODE HERE
def find_snuke(H, W, grid):
    target = "snuke"
    directions = [
        (0, 1),  # right
        (1, 0),  # down
        (1, 1),  # down-right
        (1, -1)  # down-left
    ]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for d in directions:
                    found = True
                    positions = [(i+1, j+1)]
                    for k in range(1, 5):
                        ni, nj = i + k * d[0], j + k * d[1]
                        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == target[k]:
                            positions.append((ni+1, nj+1))
                        else:
                            found = False
                            break
                    if found:
                        return positions

# Read input
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:]

# Find the positions
positions = find_snuke(H, W, grid)

# Print the result
for pos in positions:
    print(pos[0], pos[1])