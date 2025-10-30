def collect_outer_squares(grid, N):
    outer = []
    # First row: row=0, cols=0 to N-1
    outer.extend(grid[0])
    # Last column: col=N-1, rows=1 to N-1
    for row in range(1, N):
        outer.append(grid[row][N-1])
    # Last row: row=N-1, cols=N-2 to 0
    outer.extend(reversed(grid[N-1][0:N-1]))
    # First column: col=0, rows=N-2 to 1
    for row in range(N-2, 0, -1):
        outer.append(grid[row][0])
    return outer

def assign_outer_squares(grid, N, shifted):
    index = 0
    # First row: row=0, cols=0 to N-1
    for col in range(N):
        grid[0][col] = shifted[index]
        index += 1
    # Last column: col=N-1, rows=1 to N-1
    for row in range(1, N):
        grid[row][N-1] = shifted[index]
        index += 1
    # Last row: row=N-1, cols=N-2 to 0
    for col in range(N-2, -1, -1):
        grid[N-1][col] = shifted[index]
        index += 1
    # First column: col=0, rows=N-2 to 1
    for row in range(N-2, 0, -1):
        grid[row][0] = shifted[index]
        index += 1
    return grid

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    grid = [list(data[i+1].strip()) for i in range(N)]
    
    outer = collect_outer_squares(grid, N)
    shifted = [outer[-1]] + outer[:-1]
    grid = assign_outer_squares(grid, N, shifted)
    
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()