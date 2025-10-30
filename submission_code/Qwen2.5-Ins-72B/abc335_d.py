def solve(N):
    # Initialize the grid with zeros
    grid = [[0 for _ in range(N)] for _ in range(N)]
    
    # Place Takahashi at the center
    center = (N + 1) // 2 - 1
    grid[center][center] = 'T'
    
    # Directions for moving in the grid (right, down, left, up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Start from the center and move in a spiral pattern
    x, y = center, center - 1
    part = 1
    
    while part < N * N:
        for dx, dy in directions:
            while 0 <= x + dx < N and 0 <= y + dy < N and grid[x + dx][y + dy] == 0:
                x, y = x + dx, y + dy
                grid[x][y] = part
                part += 1
    
    # Print the grid
    for row in grid:
        print(' '.join(map(str, row)))

# Read input
N = int(input())
solve(N)