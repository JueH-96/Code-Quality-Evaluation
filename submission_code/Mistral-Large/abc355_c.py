import sys

def check_bingo(grid, N):
    # Check rows
    for row in grid:
        if all(row):
            return True

    # Check columns
    for col in range(N):
        if all(grid[row][col] for row in range(N)):
            return True

    # Check diagonals
    if all(grid[i][i] for i in range(N)):
        return True
    if all(grid[i][N-1-i] for i in range(N)):
        return True

    return False

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:]))

    grid = [[False] * N for _ in range(N)]

    for turn, value in enumerate(A, start=1):
        row = (value - 1) // N
        col = (value - 1) % N
        grid[row][col] = True

        if check_bingo(grid, N):
            print(turn)
            return

    print(-1)

if __name__ == "__main__":
    main()