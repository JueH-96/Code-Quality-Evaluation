import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:]))

    # Counters for rows, columns, and the two diagonals
    row_cnt = [0] * N
    col_cnt = [0] * N
    main_diag = 0
    anti_diag = 0

    for turn, x in enumerate(A, start=1):
        # Compute 0-based row i and column j
        x -= 1
        i = x // N
        j = x % N

        # Mark the cell
        row_cnt[i] += 1
        col_cnt[j] += 1
        if i == j:
            main_diag += 1
        if i + j == N - 1:
            anti_diag += 1

        # Check for Bingo
        if row_cnt[i] == N or col_cnt[j] == N or main_diag == N or anti_diag == N:
            print(turn)
            return

    # If no Bingo by the end of T turns
    print(-1)

if __name__ == "__main__":
    main()