def main():
    import sys
    data = sys.stdin.read().split()
    A, B = map(int, data)
    # Convert number to its row and column (0-indexed) in the 3x3 board.
    row_A, col_A = (A - 1) // 3, (A - 1) % 3
    row_B, col_B = (B - 1) // 3, (B - 1) % 3

    # Check if they are on the same row and horizontally adjacent (B is exactly one column to the right of A).
    if row_A == row_B and col_B - col_A == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()