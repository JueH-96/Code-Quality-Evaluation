# YOUR CODE HERE
def shift_outer_clockwise(grid, N):
    # Extract the outer layer in clockwise order
    outer_layer = []
    
    # Top row
    for j in range(N):
        outer_layer.append(grid[0][j])
    
    # Right column
    for i in range(1, N):
        outer_layer.append(grid[i][N-1])
    
    # Bottom row
    for j in range(N-2, -1, -1):
        outer_layer.append(grid[N-1][j])
    
    # Left column
    for i in range(N-2, 0, -1):
        outer_layer.append(grid[i][0])
    
    # Shift the outer layer clockwise by one position
    outer_layer = [outer_layer[-1]] + outer_layer[:-1]
    
    # Place the shifted values back into the grid
    index = 0
    
    # Top row
    for j in range(N):
        grid[0][j] = outer_layer[index]
        index += 1
    
    # Right column
    for i in range(1, N):
        grid[i][N-1] = outer_layer[index]
        index += 1
    
    # Bottom row
    for j in range(N-2, -1, -1):
        grid[N-1][j] = outer_layer[index]
        index += 1
    
    # Left column
    for i in range(N-2, 0, -1):
        grid[i][0] = outer_layer[index]
        index += 1
    
    return grid

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
grid = [list(data[i+1]) for i in range(N)]

# Shift the outer layer
shifted_grid = shift_outer_clockwise(grid, N)

# Print the resulting grid
for row in shifted_grid:
    print("".join(row))