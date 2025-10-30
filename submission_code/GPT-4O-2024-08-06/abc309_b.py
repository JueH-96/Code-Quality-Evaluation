# YOUR CODE HERE
def shift_outer_clockwise(grid, N):
    # Extract the outer layer in clockwise order
    outer_layer = []
    
    # Top row
    for j in range(N):
        outer_layer.append(grid[0][j])
    
    # Right column (excluding top and bottom corners)
    for i in range(1, N-1):
        outer_layer.append(grid[i][N-1])
    
    # Bottom row (reversed)
    for j in range(N-1, -1, -1):
        outer_layer.append(grid[N-1][j])
    
    # Left column (excluding top and bottom corners, reversed)
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
    
    # Right column (excluding top and bottom corners)
    for i in range(1, N-1):
        grid[i][N-1] = outer_layer[index]
        index += 1
    
    # Bottom row (reversed)
    for j in range(N-1, -1, -1):
        grid[N-1][j] = outer_layer[index]
        index += 1
    
    # Left column (excluding top and bottom corners, reversed)
    for i in range(N-2, 0, -1):
        grid[i][0] = outer_layer[index]
        index += 1
    
    return grid

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = [list(data[i + 1]) for i in range(N)]
    
    # Shift the outer layer clockwise
    shifted_grid = shift_outer_clockwise(grid, N)
    
    # Print the resulting grid
    for row in shifted_grid:
        print(''.join(row))

main()