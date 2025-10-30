# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # Create an N x N grid initialized with zeros
    grid = [[0] * N for _ in range(N)]
    
    # Calculate the center position
    center = N // 2
    
    # Place Takahashi at the center
    grid[center][center] = 'T'
    
    # Directions for spiral: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Start filling from the center
    x, y = center, center
    current_number = 1
    
    # Fill the grid in a spiral pattern
    for layer in range(1, (N // 2) + 1):
        # For each layer, we have four sides to fill
        for direction in directions:
            dx, dy = direction
            # Determine the number of steps to take in this direction
            steps = 2 * layer if direction in [(0, 1), (0, -1)] else 2 * layer - 1
            for _ in range(steps):
                # Move to the next cell in the current direction
                x += dx
                y += dy
                # Place the current number
                grid[x][y] = current_number
                current_number += 1
    
    # Fill the remaining numbers in the outermost layer
    # This is necessary because the last layer might not be fully filled
    for direction in directions:
        dx, dy = direction
        steps = N - 1
        for _ in range(steps):
            x += dx
            y += dy
            if grid[x][y] == 0:
                grid[x][y] = current_number
                current_number += 1
    
    # Print the grid
    for row in grid:
        print(' '.join(map(str, row)))