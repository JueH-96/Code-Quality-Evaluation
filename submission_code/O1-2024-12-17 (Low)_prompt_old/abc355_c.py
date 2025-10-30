def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, T = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Arrays to store how many cells in each row/column are marked
    row_count = [0] * N
    col_count = [0] * N

    # Variables to store how many cells of each diagonal are marked
    main_diag_count = 0
    anti_diag_count = 0

    # Process each announced integer in order
    for turn, value in enumerate(A, start=1):
        # Convert the announced number to 0-based row and column
        value -= 1
        r, c = value // N, value % N

        # Mark the row and column
        row_count[r] += 1
        if row_count[r] == N:
            print(turn)
            return
        col_count[c] += 1
        if col_count[c] == N:
            print(turn)
            return

        # If it lies on the main diagonal
        if r == c:
            main_diag_count += 1
            if main_diag_count == N:
                print(turn)
                return

        # If it lies on the anti-diagonal
        if r + c == N - 1:
            anti_diag_count += 1
            if anti_diag_count == N:
                print(turn)
                return

    # If we finish all turns without Bingo, print -1
    print(-1)

def solve_wrapper():
    solve()

# Call the solve function
if __name__ == "__main__":
    solve()