def main():
    import sys
    data = sys.stdin.read().splitlines()
    board = data[:8]  # there are exactly 8 lines for an 8x8 board

    # A square is safe only if its row and column are both free of any pieces.
    # Count safe rows: rows which contain no '#' character.
    safe_rows = sum(1 for row in board if '#' not in row)

    # Count safe columns: columns where no row has a '#' placed.
    safe_cols = 0
    for col in range(8):
        col_safe = True
        for row in board:
            if row[col] == '#':
                col_safe = False
                break
        if col_safe:
            safe_cols += 1

    # The number of safe placements is a product of the number of safe rows and safe columns
    print(safe_rows * safe_cols)

if __name__ == '__main__':
    main()