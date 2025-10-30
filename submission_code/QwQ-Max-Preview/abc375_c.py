import sys

def main():
    N = int(sys.stdin.readline())
    grid = [list(sys.stdin.readline().strip()) for _ in range(N)]
    
    for i in range(1, N//2 + 1):
        start = i - 1
        end = N - i
        new_grid = [row.copy() for row in grid]
        for x in range(start, end + 1):
            for y in range(start, end + 1):
                # Convert to problem's 1-based coordinates
                # Then compute new coordinates
                new_x = y
                new_y = N - x - 1
                new_grid[new_x][new_y] = grid[x][y]
        grid = new_grid
    
    for row in grid:
        print(''.join(row))

if __name__ == '__main__':
    main()