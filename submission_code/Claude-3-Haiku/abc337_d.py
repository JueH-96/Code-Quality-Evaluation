from typing import List

def solve(H: int, W: int, K: int, grid: List[str]) -> int:
    # Initialize the grid as a 2D list
    grid_2d = [list(row) for row in grid]

    # Function to check if a horizontal sequence of K cells with 'o' exists
    def check_horizontal(i, j):
        for k in range(j, j + K):
            if k >= W or grid_2d[i][k] != 'o':
                return False
        return True

    # Function to check if a vertical sequence of K cells with 'o' exists
    def check_vertical(i, j):
        for k in range(i, i + K):
            if k >= H or grid_2d[k][j] != 'o':
                return False
        return True

    # Count the number of operations required
    operations = 0
    for i in range(H):
        for j in range(W):
            if grid_2d[i][j] == '.':
                grid_2d[i][j] = 'o'
                operations += 1
                if check_horizontal(i, j) or check_vertical(i, j):
                    return operations
                grid_2d[i][j] = '.'

    # If no sequence of K cells with 'o' can be formed, return -1
    return -1