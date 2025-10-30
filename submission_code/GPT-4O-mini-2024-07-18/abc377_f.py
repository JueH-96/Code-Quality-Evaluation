def count_safe_squares(N, M, pieces):
    # Sets to track attacked rows, columns, and diagonals
    attacked_rows = set()
    attacked_columns = set()
    attacked_diagonal1 = set()  # i + j
    attacked_diagonal2 = set()  # i - j

    # Populate the sets with the positions of the existing pieces
    for a, b in pieces:
        attacked_rows.add(a)
        attacked_columns.add(b)
        attacked_diagonal1.add(a + b)
        attacked_diagonal2.add(a - b)

    # Total squares in the grid
    total_squares = N * N

    # Calculate the number of attacked squares
    attacked_squares = 0

    # Count attacked squares in rows
    attacked_squares += len(attacked_rows) * N

    # Count attacked squares in columns
    attacked_squares += len(attacked_columns) * N

    # Count attacked squares in diagonals
    # For diagonal i + j = k, there are min(k, 2N - k) squares
    for d1 in attacked_diagonal1:
        if 2 <= d1 <= 2 * N:
            attacked_squares += min(d1 - 1, 2 * N - d1)

    # For diagonal i - j = k, there are min(N + k, N - k) squares
    for d2 in attacked_diagonal2:
        if -N + 1 <= d2 <= N - 1:
            attacked_squares += min(N + d2, N - d2)

    # Each attacked square has been counted multiple times, we need to subtract overlaps
    # Each piece occupies one square, so we need to subtract M
    attacked_squares -= M

    # The number of safe squares is total squares minus attacked squares
    safe_squares = total_squares - attacked_squares

    return safe_squares

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, M = map(int, data[0].split())
    pieces = [tuple(map(int, line.split())) for line in data[1:M + 1]]
    
    result = count_safe_squares(N, M, pieces)
    print(result)

if __name__ == "__main__":
    main()