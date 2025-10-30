import sys

def solve(n, r, c):
    # Initialize the grid with empty cells
    grid = [['.' for _ in range(n)] for _ in range(n)]

    # Fill the grid based on the given conditions
    for i in range(n):
        for j in range(n):
            if i == 0:
                grid[i][j] = c[j]
            elif j == 0:
                grid[i][j] = r[i]

    # Check if the grid can be filled
    for i in range(n):
        for j in range(n):
            if grid[i][j] == '.':
                for char in 'ABC':
                    if char not in [grid[i][k] for k in range(n)] and char not in [grid[k][j] for k in range(n)]:
                        grid[i][j] = char
                        break

    # Check if the grid is valid
    for i in range(n):
        if len(set([grid[i][j] for j in range(n)])) != 3 or len(set([grid[j][i] for j in range(n)])) != 3:
            return "No"

    return "Yes
" + "
".join("".join(row) for row in grid)

# Read the inputs from stdin
n = int(sys.stdin.readline().strip())
r = sys.stdin.readline().strip()
c = sys.stdin.readline().strip()

# Solve the problem and write the answer to stdout
print(solve(n, r, c))