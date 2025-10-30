def generate_pattern(N):
    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    for i in range(1, N + 1):
        j = N + 1 - i
        if i <= j:
            # Determine the color
            color = '#' if i % 2 == 1 else '.'
            # Fill the rectangle from (i-1, i-1) to (j-1, j-1)
            for row in range(i-1, j):
                for col in range(i-1, j):
                    grid[row][col] = color
    
    # Print the grid
    for row in grid:
        print(''.join(row))

import sys
input = sys.stdin.read
N = int(input().strip())
generate_pattern(N)