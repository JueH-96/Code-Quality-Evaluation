def main():
    import sys
    input = sys.stdin.readline
    grid = [input().rstrip() for _ in range(8)]
    
    # Count rows with no existing piece
    empty_rows = sum(1 for row in grid if '#' not in row)
    
    # Count columns with no existing piece
    empty_cols = 0
    for c in range(8):
        if all(grid[r][c] == '.' for r in range(8)):
            empty_cols += 1
    
    # Any intersection of an empty row and empty column
    # is an empty cell not attacked by any piece
    print(empty_rows * empty_cols)

if __name__ == "__main__":
    main()