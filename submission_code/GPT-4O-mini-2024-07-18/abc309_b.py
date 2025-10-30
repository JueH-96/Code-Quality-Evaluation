def shift_outer_squares_clockwise(N, grid):
    # Extract the outer squares in clockwise order
    outer_squares = []
    
    # Top row
    for j in range(N):
        outer_squares.append(grid[0][j])
    
    # Right column (excluding the first and last element)
    for i in range(1, N - 1):
        outer_squares.append(grid[i][N - 1])
    
    # Bottom row (in reverse order)
    for j in range(N - 1, -1, -1):
        outer_squares.append(grid[N - 1][j])
    
    # Left column (in reverse order, excluding the first and last element)
    for i in range(N - 2, 0, -1):
        outer_squares.append(grid[i][0])
    
    # Shift the outer squares clockwise by one position
    shifted_squares = [outer_squares[-1]] + outer_squares[:-1]
    
    # Fill the new grid with the shifted outer squares
    new_grid = [[0] * N for _ in range(N)]
    
    index = 0
    
    # Top row
    for j in range(N):
        new_grid[0][j] = shifted_squares[index]
        index += 1
    
    # Right column
    for i in range(1, N - 1):
        new_grid[i][N - 1] = shifted_squares[index]
        index += 1
    
    # Bottom row
    for j in range(N - 1, -1, -1):
        new_grid[N - 1][j] = shifted_squares[index]
        index += 1
    
    # Left column
    for i in range(N - 2, 0, -1):
        new_grid[i][0] = shifted_squares[index]
        index += 1
    
    return new_grid

import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    grid = [list(data[i + 1]) for i in range(N)]
    
    new_grid = shift_outer_squares_clockwise(N, grid)
    
    for row in new_grid:
        print(''.join(row))

if __name__ == "__main__":
    main()