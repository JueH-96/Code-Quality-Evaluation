def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, T = map(int, data[:2])
    announced = list(map(int, data[2:]))

    # We will track how many marks are in each row and each column
    row_count = [0] * (N + 1)
    col_count = [0] * (N + 1)

    main_diag_count = 0      # For the top-left to bottom-right diagonal
    anti_diag_count = 0      # For the top-right to bottom-left diagonal

    # Process each turn
    for turn_index, value in enumerate(announced, start=1):
        # Convert the value to its (row, col) in 1-based indexing
        r = (value - 1) // N + 1
        c = (value - 1) % N + 1
        
        # Mark the row and the column
        row_count[r] += 1
        col_count[c] += 1
        
        # If it's on the main diagonal
        if r == c:
            main_diag_count += 1
        
        # If it's on the anti-diagonal (top-right to bottom-left)
        if r + c == N + 1:
            anti_diag_count += 1
        
        # Check if we have Bingo
        if (
            row_count[r] == N
            or col_count[c] == N
            or main_diag_count == N
            or anti_diag_count == N
        ):
            print(turn_index)
            return

    # If we could not get Bingo within T turns
    print(-1)

# Do not forget to call main()
if __name__ == "__main__":
    main()