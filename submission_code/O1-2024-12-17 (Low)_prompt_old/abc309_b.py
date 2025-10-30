def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = data[1:]
    
    # Convert each row string into a list of integers
    grid = [list(map(int, row)) for row in A]
    
    # Collect coordinates of the outer layer in clockwise order
    outer_coords = []
    
    # Top row (left to right): (0,0) to (0, N-1)
    for j in range(N):
        outer_coords.append((0, j))
    
    # Right column (top to bottom): (1, N-1) to (N-2, N-1)
    for i in range(1, N - 1):
        outer_coords.append((i, N - 1))
    
    # Bottom row (right to left): (N-1, N-1) to (N-1, 0)
    if N > 1:
        for j in range(N - 1, -1, -1):
            outer_coords.append((N - 1, j))
    
    # Left column (bottom to top): (N-2, 0) to (1, 0)
    if N > 1:
        for i in range(N - 2, 0, -1):
            outer_coords.append((i, 0))
    
    # Extract the values in the outer layer
    outer_values = [grid[r][c] for (r, c) in outer_coords]
    
    # Shift the values by 1 position to the right (clockwise)
    # So the last goes to the front
    outer_values = [outer_values[-1]] + outer_values[:-1]
    
    # Place the shifted values back into the grid
    for (r, c), val in zip(outer_coords, outer_values):
        grid[r][c] = val
    
    # Print the resulting grid
    for row in grid:
        print("".join(map(str, row)))

# Let's call solve()
solve()