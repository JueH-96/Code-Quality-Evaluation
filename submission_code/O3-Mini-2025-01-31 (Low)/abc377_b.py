def main():
    import sys
    board = [line.strip() for line in sys.stdin.readlines() if line.strip()]
    
    # Identify safe rows (rows without any piece)
    safe_rows = [i for i in range(8) if '#' not in board[i]]
    
    # Identify safe columns (columns without any piece). For each col j, check if any row i has '#'
    safe_cols = []
    for j in range(8):
        col_has_piece = any(board[i][j] == '#' for i in range(8))
        if not col_has_piece:
            safe_cols.append(j)
    
    count = 0
    for i in safe_rows:
        for j in safe_cols:
            # Square (i, j) must be empty
            if board[i][j] == '.':
                count += 1

    print(count)

if __name__ == '__main__':
    main()