# YOUR CODE HERE
def rotate_grid(grid, n):
    for i in range(1, n // 2 + 1):
        for x in range(i, n - i + 1):
            for y in range(i, n - i + 1):
                grid[y][n - x] = grid[x][y]
    return grid

def main():
    n = int(input())
    grid = [list(input().strip()) for _ in range(n)]
    
    rotated_grid = rotate_grid(grid, n)
    
    for row in rotated_grid:
        print(''.join(row))

if __name__ == "__main__":
    main()