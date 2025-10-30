def create_spiral(N):
    # Initialize the grid with zeros
    grid = [[0 for _ in range(N)] for _ in range(N)]
    
    # Define the center position for Takahashi
    center = (N - 1) // 2
    
    # Place Takahashi at the center
    grid[center][center] = 'T'
    
    # Define the directions for the spiral (right, down, left, up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Start from the center and move left to start the spiral
    x, y = center, center - 1
    
    # Initialize the part number and direction index
    part_num = 1
    dir_idx = 0
    
    # Loop to fill the grid with dragon parts
    for _ in range(N * N - 1):
        # Place the current part number
        grid[x][y] = part_num
        part_num += 1
        
        # Calculate the next cell
        next_x, next_y = x + directions[dir_idx][0], y + directions[dir_idx][1]
        
        # Check if the next cell is within bounds and not already filled
        if 0 <= next_x < N and 0 <= next_y < N and grid[next_x][next_y] == 0:
            # Move to the next cell
            x, y = next_x, next_y
        else:
            # Change direction (right -> down -> left -> up -> right ...)
            dir_idx = (dir_idx + 1) % 4
            x += directions[dir_idx][0]
            y += directions[dir_idx][1]
    
    return grid

# Read N from stdin
N = int(input().strip())

# Create the spiral grid
spiral_grid = create_spiral(N)

# Print the grid
for row in spiral_grid:
    print(' '.join(str(cell) for cell in row))