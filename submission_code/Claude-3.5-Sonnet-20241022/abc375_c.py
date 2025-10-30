def transform_grid(N, grid):
    # Convert grid to list of lists for easier manipulation
    grid = [list(row) for row in grid]
    
    # Process for N/2 iterations
    for i in range(1, N//2 + 1):
        # Create a copy of the grid for simultaneous updates
        new_grid = [row[:] for row in grid]
        
        # For all pairs of x,y between i and N+1-i
        for x in range(i, N+1-i+1):
            for y in range(i, N+1-i+1):
                # Replace color of cell (y, N+1-x) with color of cell (x,y)
                new_grid[y-1][N-x] = grid[x-1][y-1]
        
        # Update the grid
        grid = new_grid
    
    return grid

def main():
    # Read input
    N = int(input())
    grid = []
    for _ in range(N):
        row = input().strip()
        grid.append(row)
    
    # Transform the grid
    result = transform_grid(N, grid)
    
    # Print output
    for row in result:
        print(''.join(row))

if __name__ == "__main__":
    main()