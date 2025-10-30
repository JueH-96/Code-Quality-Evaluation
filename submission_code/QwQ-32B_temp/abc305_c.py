import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Find rows with at least one '#'
    rows_with_hash = [i for i in range(H) if '#' in grid[i]]
    a = min(rows_with_hash)
    b = max(rows_with_hash)
    
    # Find columns with at least one '#'
    columns_with_hash = []
    for j in range(W):
        if any(grid[i][j] == '#' for i in range(H)):
            columns_with_hash.append(j)
    c = min(columns_with_hash)
    d = max(columns_with_hash)
    
    # Search for the missing '.' in the rectangle
    for i in range(a, b + 1):
        for j in range(c, d + 1):
            if grid[i][j] == '.':
                print(i + 1, j + 1)
                return

if __name__ == "__main__":
    main()