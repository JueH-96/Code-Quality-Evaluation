def main():
    grid = [input().strip() for _ in range(8)]
    # Determine which rows and columns already have pieces
    row_has_piece = [('#' in row) for row in grid]
    col_has_piece = []
    for c in range(8):
        has = False
        for r in range(8):
            if grid[r][c] == '#':
                has = True
                break
        col_has_piece.append(has)

    # Count empty squares (i,j) such that row i and column j have no pieces
    ans = 0
    for i in range(8):
        for j in range(8):
            if grid[i][j] == '.' and not row_has_piece[i] and not col_has_piece[j]:
                ans += 1

    print(ans)

main()