import sys

def main():
    import sys
    N_and_rest = sys.stdin.read().splitlines()
    N = int(N_and_rest[0])
    grid = [list(line) for line in N_and_rest[1:N+1]]
    
    for i in range(1, N//2 + 1):
        grid_copy = [row[:] for row in grid]
        for x in range(i-1, N - i + 1):
            for y in range(i-1, N - i + 1):
                grid_copy[y][N - 1 - x] = grid[x][y]
        grid = grid_copy
    
    for row in grid:
        print(''.join(row))

if __name__ == '__main__':
    main()