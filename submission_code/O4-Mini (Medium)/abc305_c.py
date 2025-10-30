import sys

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    
    # Find the bounding rows that still contain at least one '#'
    rows_with_hash = [i for i in range(H) if '#' in grid[i]]
    min_row = min(rows_with_hash)
    max_row = max(rows_with_hash)
    
    # Find the bounding columns that still contain at least one '#'
    cols_with_hash = []
    for j in range(W):
        for i in range(H):
            if grid[i][j] == '#':
                cols_with_hash.append(j)
                break
    min_col = min(cols_with_hash)
    max_col = max(cols_with_hash)
    
    # Within that rectangle, find the single missing cookie ('.')
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                # Output in 1-based indexing
                print(i + 1, j + 1)
                return

if __name__ == "__main__":
    main()