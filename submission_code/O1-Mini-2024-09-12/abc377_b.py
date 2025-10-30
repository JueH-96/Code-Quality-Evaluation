# YOUR CODE HERE
def solve():
    grid = [input().strip() for _ in range(8)]
    
    # Count rows without any pieces
    rows_no_piece = sum(1 for row in grid if '#' not in row)
    
    # Count columns without any pieces
    cols_no_piece = 0
    for j in range(8):
        has_piece = False
        for i in range(8):
            if grid[i][j] == '#':
                has_piece = True
                break
        if not has_piece:
            cols_no_piece +=1
    
    # The number of safe squares is the product of rows_no_piece and cols_no_piece
    print(rows_no_piece * cols_no_piece)