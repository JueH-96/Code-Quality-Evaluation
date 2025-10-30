def main():
    n = int(input().strip())
    grid = [list(input().strip()) for _ in range(n)]
    
    for i in range(1, n//2 + 1):
        grid = apply_operation(grid, n, i)
    
    for row in grid:
        print(''.join(row))

def apply_operation(grid, n, i):
    """
    Apply the operation for the given i value.
    
    For all pairs of integers x, y between i and N+1-i, inclusive,
    replace the color of cell (y, N+1-x) with the color of cell (x, y).
    """
    new_grid = [row[:] for row in grid]  # Create a deep copy
    
    for x in range(i, n+1-i+1):
        for y in range(i, n+1-i+1):
            new_grid[y-1][n-x] = grid[x-1][y-1]  # Adjusted for 0-indexed arrays
    
    return new_grid

if __name__ == "__main__":
    main()