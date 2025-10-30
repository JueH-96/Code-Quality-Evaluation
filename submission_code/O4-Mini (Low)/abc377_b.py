def main():
    import sys

    # Read the 8x8 board
    grid = [sys.stdin.readline().rstrip("
") for _ in range(8)]

    # Determine which rows and columns already contain a piece ('#')
    row_has_piece = [False] * 8
    col_has_piece = [False] * 8

    for i in range(8):
        for j in range(8):
            if grid[i][j] == '#':
                row_has_piece[i] = True
                col_has_piece[j] = True

    # Count empty squares that are neither in a row nor in a column with an existing piece
    safe_count = 0
    for i in range(8):
        for j in range(8):
            if grid[i][j] == '.' and not row_has_piece[i] and not col_has_piece[j]:
                safe_count += 1

    print(safe_count)

if __name__ == "__main__":
    main()