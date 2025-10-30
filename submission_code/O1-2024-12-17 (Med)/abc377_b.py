def main():
    # Read the input lines
    board = [input().strip() for _ in range(8)]
    
    # Arrays to keep track of whether a row or a column has a piece
    row_has_piece = [False] * 8
    col_has_piece = [False] * 8
    
    # Mark rows and columns that contain pieces
    for r in range(8):
        for c in range(8):
            if board[r][c] == '#':
                row_has_piece[r] = True
                col_has_piece[c] = True
    
    # Count squares that are empty and not in a row or column with a piece
    count = 0
    for r in range(8):
        for c in range(8):
            if board[r][c] == '.' and not row_has_piece[r] and not col_has_piece[c]:
                count += 1
    
    print(count)

# Call main() to run the solution
main()