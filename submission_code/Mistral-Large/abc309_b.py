import sys

def rotate_outer_grid(N, grid):
    # Save the outer elements
    top_row = grid[0][:]
    bottom_row = grid[N-1][:]
    left_col = [grid[i][0] for i in range(N)]
    right_col = [grid[i][N-1] for i in range(N)]

    # Rotate the outer elements
    grid[0][1:] = top_row[:N-1]
    for i in range(1, N-1):
        grid[i][N-1] = right_col[i-1]
    grid[N-1][:N-1] = bottom_row[1:]
    for i in range(1, N-1):
        grid[i][0] = left_col[i+1]
    grid[0][0] = left_col[1]
    grid[N-1][N-1] = right_col[N-2]
    grid[N-1][0] = bottom_row[0]
    grid[0][N-1] = top_row[N-2]

    return grid

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    grid = []
    index = 1
    for i in range(N):
        row = list(map(int, data[index]))
        grid.append(row)
        index += 1

    result_grid = rotate_outer_grid(N, grid)

    for row in result_grid:
        print(''.join(map(str, row)))

if __name__ == "__main__":
    main()