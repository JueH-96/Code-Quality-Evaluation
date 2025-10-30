# YOUR CODE HERE
import sys

def shift_outer_squares(grid, N):
    # Extract the outer squares in clockwise order
    outer_squares = []
    
    # Top row
    for j in range(N):
        outer_squares.append(grid[0][j])
    
    # Right column (excluding the top-right corner)
    for i in range(1, N):
        outer_squares.append(grid[i][N-1])
    
    # Bottom row (excluding the bottom-right corner)
    for j in range(N-2, -1, -1):
        outer_squares.append(grid[N-1][j])
    
    # Left column (excluding the bottom-left and top-left corners)
    for i in range(N-2, 0, -1):
        outer_squares.append(grid[i][0])
    
    # Shift the outer squares clockwise by one position
    outer_squares = [outer_squares[-1]] + outer_squares[:-1]
    
    # Place the shifted values back into the grid
    index = 0
    
    # Top row
    for j in range(N):
        grid[0][j] = outer_squares[index]
        index += 1
    
    # Right column (excluding the top-right corner)
    for i in range(1, N):
        grid[i][N-1] = outer_squares[index]
        index += 1
    
    # Bottom row (excluding the bottom-right corner)
    for j in range(N-2, -1, -1):
        grid[N-1][j] = outer_squares[index]
        index += 1
    
    # Left column (excluding the bottom-left and top-left corners)
    for i in range(N-2, 0, -1):
        grid[i][0] = outer_squares[index]
        index += 1

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    grid = []
    index = 1
    for i in range(N):
        row = list(map(int, input[index]))
        grid.append(row)
        index += 1
    
    shift_outer_squares(grid, N)
    
    for row in grid:
        print(''.join(map(str, row)))

if __name__ == "__main__":
    main()