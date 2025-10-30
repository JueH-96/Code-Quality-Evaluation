import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:]))

    # Counters for rows, columns, and the two diagonals
    row_counts = [0] * N
    col_counts = [0] * N
    diag1 = 0  # top-left to bottom-right
    diag2 = 0  # top-right to bottom-left

    for turn_idx, a in enumerate(A, start=1):
        # Compute 0-based row and column from the number a
        r = (a - 1) // N
        c = (a - 1) % N

        # Mark this cell
        row_counts[r] += 1
        col_counts[c] += 1
        if r == c:
            diag1 += 1
        if r + c == N - 1:
            diag2 += 1

        # Check for Bingo
        if row_counts[r] == N or col_counts[c] == N or diag1 == N or diag2 == N:
            print(turn_idx)
            return

    # If no Bingo was achieved
    print(-1)

if __name__ == "__main__":
    main()