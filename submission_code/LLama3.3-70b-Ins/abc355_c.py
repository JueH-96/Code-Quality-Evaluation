def check_bingo(grid, N):
    # Check rows
    for row in grid:
        if all(cell for cell in row):
            return True

    # Check columns
    for col in range(N):
        if all(grid[row][col] for row in range(N)):
            return True

    # Check diagonals
    if all(grid[i][i] for i in range(N)):
        return True
    if all(grid[i][N - i - 1] for i in range(N)):
        return True

    return False


def main():
    N, T = map(int, input().split())
    A = list(map(int, input().split()))

    grid = [[False] * N for _ in range(N)]

    for turn, num in enumerate(A, start=1):
        row = (num - 1) // N
        col = (num - 1) % N
        grid[row][col] = True

        if check_bingo(grid, N):
            print(turn)
            return

    print(-1)


if __name__ == "__main__":
    main()