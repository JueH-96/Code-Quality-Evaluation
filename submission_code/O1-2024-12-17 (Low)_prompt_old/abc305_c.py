def solve():
    import sys
    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [input().rstrip('
') for _ in range(H)]

    # Collect all coordinates of squares that currently have a cookie
    cookie_positions = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == '#']

    # Determine the bounding rectangle of all current cookies
    row_min = min(r for r, _ in cookie_positions)
    row_max = max(r for r, _ in cookie_positions)
    col_min = min(c for _, c in cookie_positions)
    col_max = max(c for _, c in cookie_positions)

    # Within this bounding rectangle, exactly one position should be a missing cookie ('.')
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            if grid[r][c] == '.':
                print(r + 1, c + 1)
                return

# Call solve() to read input, process, and print output
def main():
    solve()

if __name__ == "__main__":
    main()