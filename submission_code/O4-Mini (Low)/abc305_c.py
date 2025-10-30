def main():
    import sys
    input = sys.stdin.readline
    
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    
    # Find the minimal enclosing rectangle of all remaining '#'
    row_min = H
    row_max = -1
    col_min = W
    col_max = -1
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < row_min: row_min = i
                if i > row_max: row_max = i
                if j < col_min: col_min = j
                if j > col_max: col_max = j
    
    # Within that rectangle, find the one missing cookie ('.')
    for i in range(row_min, row_max + 1):
        for j in range(col_min, col_max + 1):
            if grid[i][j] == '.':
                # convert to 1-based coordinates
                print(i + 1, j + 1)
                return

if __name__ == "__main__":
    main()