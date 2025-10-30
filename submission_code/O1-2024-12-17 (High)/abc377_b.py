def main():
    board = [input().strip() for _ in range(8)]
    
    rows_with_piece = set()
    cols_with_piece = set()
    
    # Identify all rows and columns that contain at least one piece.
    for i in range(8):
        for j in range(8):
            if board[i][j] == '#':
                rows_with_piece.add(i)
                cols_with_piece.add(j)
    
    # Count empty squares ('.') that are in rows and columns with no existing pieces.
    count = 0
    for i in range(8):
        if i not in rows_with_piece:
            for j in range(8):
                if j not in cols_with_piece and board[i][j] == '.':
                    count += 1
    
    print(count)

# Do not forget to call main
if __name__ == "__main__":
    main()