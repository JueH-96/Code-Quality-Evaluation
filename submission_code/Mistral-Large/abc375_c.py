import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = data[1:]

    grid = [list(A[i]) for i in range(N)]

    for i in range(1, N // 2 + 1):
        new_grid = [row[:] for row in grid]
        for x in range(i, N + 1 - i):
            for y in range(i, N + 1 - i):
                new_grid[y][N - x] = grid[x][y]
        grid = new_grid

    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()