# YOUR CODE HERE
import sys
input = sys.stdin.read

def rotate_grid(N, grid):
    # Create a new grid to store the result
    new_grid = [['' for _ in range(N)] for _ in range(N)]
    
    # Perform the operations
    for i in range(1, N//2 + 1):
        for x in range(i, N + 1 - i + 1):
            for y in range(i, N + 1 - i + 1):
                new_grid[y-1][N-x] = grid[x-1][y-1]
    
    # Fill in the remaining cells that were not affected by the operation
    for i in range(N):
        for j in range(N):
            if new_grid[i][j] == '':
                new_grid[i][j] = grid[i][j]
    
    return new_grid

def main():
    data = input().split()
    N = int(data[0])
    grid = [list(data[i + 1]) for i in range(N)]
    
    result_grid = rotate_grid(N, grid)
    
    for row in result_grid:
        print(''.join(row))

if __name__ == "__main__":
    main()