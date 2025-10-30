def generate_grid(N):
    # Initialize grid
    grid = [[None for _ in range(N)] for _ in range(N)]
    
    # Place Takahashi at the center
    center = (N - 1) // 2  # 0-indexed
    grid[center][center] = 'T'
    
    # Define the spiral direction: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir_idx = 0
    
    # Start at the top-left corner
    r, c = 0, 0
    part = 1
    
    # Fill the grid in a spiral pattern
    while part <= N**2 - 1:  # Excluding Takahashi's cell
        # Fill the current cell if it's empty
        if grid[r][c] is None:
            grid[r][c] = part
            part += 1
        
        # Find the next valid cell
        found_valid_cell = False
        for i in range(4):  # Try all four directions
            next_dir_idx = (dir_idx + i) % 4
            dr, dc = directions[next_dir_idx]
            new_r, new_c = r + dr, c + dc
            
            if 0 <= new_r < N and 0 <= new_c < N and grid[new_r][new_c] is None:
                r, c = new_r, new_c
                dir_idx = next_dir_idx
                found_valid_cell = True
                break
        
        # If no valid cell is found, break out of the loop
        if not found_valid_cell:
            break
    
    return grid

# Read input, generate grid, and print output
N = int(input())
grid = generate_grid(N)

# Output the grid
for row in grid:
    print(' '.join(map(str, row)))