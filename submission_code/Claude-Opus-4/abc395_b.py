# YOUR CODE HERE
N = int(input())

# Initialize the grid with all cells as uncolored (we'll use '?' for uncolored)
grid = [['?' for _ in range(N)] for _ in range(N)]

# Process each i from 1 to N
for i in range(1, N + 1):
    j = N + 1 - i
    
    if i <= j:
        # Fill the rectangular region from (i,i) to (j,j)
        # Note: we need to convert to 0-based indexing
        color = '#' if i % 2 == 1 else '.'
        
        for row in range(i - 1, j):
            for col in range(i - 1, j):
                grid[row][col] = color

# Print the grid
for row in grid:
    print(''.join(row))