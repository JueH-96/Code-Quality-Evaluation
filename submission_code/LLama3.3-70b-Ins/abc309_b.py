import sys

def read_input():
    """Read input from stdin."""
    lines = sys.stdin.readlines()
    n = int(lines[0].strip())
    grid = [list(map(int, line.strip())) for line in lines[1:]]
    return n, grid

def shift_outer_squares(n, grid):
    """Shift outer squares clockwise by one square each."""
    # Extract outer squares in clockwise order
    outer_squares = []
    for j in range(n):
        outer_squares.append(grid[0][j])
    for i in range(1, n):
        outer_squares.append(grid[i][n-1])
    for j in range(n-1, -1, -1):
        outer_squares.append(grid[n-1][j])
    for i in range(n-1, 0, -1):
        outer_squares.append(grid[i][0])

    # Shift outer squares clockwise by one square each
    outer_squares = outer_squares[-1:] + outer_squares[:-1]

    # Update grid with shifted outer squares
    index = 0
    for j in range(n):
        grid[0][j] = outer_squares[index]
        index += 1
    for i in range(1, n):
        grid[i][n-1] = outer_squares[index]
        index += 1
    for j in range(n-1, -1, -1):
        grid[n-1][j] = outer_squares[index]
        index += 1
    for i in range(n-1, 0, -1):
        grid[i][0] = outer_squares[index]
        index += 1

    return grid

def print_grid(grid):
    """Print grid to stdout."""
    for row in grid:
        print(''.join(map(str, row)))

def main():
    n, grid = read_input()
    grid = shift_outer_squares(n, grid)
    print_grid(grid)

if __name__ == "__main__":
    main()