def main():
    board = [input().rstrip() for _ in range(8)]

    rows_with_pieces = set()
    cols_with_pieces = set()

    # Identify which rows and columns have existing pieces
    for i in range(8):
        for j in range(8):
            if board[i][j] == '#':
                rows_with_pieces.add(i)
                cols_with_pieces.add(j)

    # Count how many empty squares are NOT in any row or column containing an existing piece
    safe_count = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == '.' and i not in rows_with_pieces and j not in cols_with_pieces:
                safe_count += 1

    print(safe_count)

# Do not remove the call to main()
main()