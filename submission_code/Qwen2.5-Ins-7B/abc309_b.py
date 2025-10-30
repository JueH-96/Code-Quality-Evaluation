# YOUR CODE HERE
n = int(input())
grid = [list(map(int, input())) for _ in range(n)]

# Extract the outer squares
outer_squares = []
for i in range(n):
    outer_squares.append(grid[0][i])
    outer_squares.append(grid[i][-1])
    outer_squares.append(grid[-1][-i-1])
    outer_squares.append(grid[-i-1][0])

# Shift the outer squares clockwise by one
shifted_squares = outer_squares[1:] + [outer_squares[0]]

# Update the grid with the shifted outer squares
k = 0
for i in range(n):
    for j in range(n):
        if (i == 0 or i == n-1) and (j == 0 or j == n-1):
            grid[i][j] = shifted_squares[k]
            k += 1

# Print the resulting grid
for row in grid:
    print(''.join(map(str, row)))