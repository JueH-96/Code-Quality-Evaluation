# YOUR CODE HERE

def generate_pattern(N):
    # Initialize the grid with empty strings
    grid = [['' for _ in range(N)] for _ in range(N)]
    
    # Iterate over each row
    for i in range(1, N + 1):
        j = N + 1 - i
        if i <= j:
            # Determine the color based on the row number
            color = '#' if i % 2 == 1 else '.'
            # Fill the rectangular region from (i, i) to (j, j)
            for x in range(i - 1, j):
                for y in range(i - 1, j):
                    grid[x][y] = color
    
    # Convert the grid to the required output format
    for row in grid:
        print(''.join(row))

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Generate and print the pattern
generate_pattern(N)