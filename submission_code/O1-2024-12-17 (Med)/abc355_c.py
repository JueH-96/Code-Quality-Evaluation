def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, T = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Counters for rows, columns, and the two diagonals
    row_count = [0]*N
    col_count = [0]*N
    diag0_count = 0
    diag1_count = 0

    for turn_index, val in enumerate(A):
        # Compute (row, col) for the value val
        val -= 1  # zero-based
        r = val // N
        c = val % N

        # Update counters
        row_count[r] += 1
        if row_count[r] == N:
            print(turn_index+1)
            return

        col_count[c] += 1
        if col_count[c] == N:
            print(turn_index+1)
            return

        if r == c:
            diag0_count += 1
            if diag0_count == N:
                print(turn_index+1)
                return

        if r + c == N - 1:
            diag1_count += 1
            if diag1_count == N:
                print(turn_index+1)
                return

    # If no bingo found
    print(-1)

# Do not forget to call main
if __name__ == "__main__":
    main()