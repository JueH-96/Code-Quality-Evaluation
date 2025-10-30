def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    H = int(data[0])
    W = int(data[1])
    grid = data[2:]
    
    # Find the boundaries of the rectangle from the positions containing '#'.
    # Even though one cookie is missing from the rectangle,
    # the remaining cookies uniquely determine the top, bottom, left, and right boundaries.
    min_row, max_row = H, 0
    min_col, max_col = W, 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < min_row:
                    min_row = i
                if i > max_row:
                    max_row = i
                if j < min_col:
                    min_col = j
                if j > max_col:
                    max_col = j
                    
    # The original rectangle was from (min_row, min_col) to (max_row, max_col) (0-indexed).
    # Now, exactly one cell inside that rectangle is missing its cookie ('.'),
    # so we find that cell.
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] != '#':
                # Answer is expected 1-indexed.
                sys.stdout.write(f"{i + 1} {j + 1}")
                return

if __name__ == '__main__':
    main()