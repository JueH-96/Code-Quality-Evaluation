def main():
    import sys
    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    # Find the bounding rectangle (min/max rows and cols) that already have a black cell ('#')
    r_min, r_max = H, -1
    c_min, c_max = W, -1
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                r_min = min(r_min, i)
                r_max = max(r_max, i)
                c_min = min(c_min, j)
                c_max = max(c_max, j)
    
    # Check that every cell inside the rectangle is not forced to be white.
    # Those cells can be painted black if they are '?', but if any is already white ('.'), it's impossible.
    for i in range(r_min, r_max + 1):
        for j in range(c_min, c_max + 1):
            if grid[i][j] == '.':
                print("No")
                return

    # Check that all cells outside the bounding rectangle are white or can be painted white.
    # If any cell is already black outside, then it's impossible to form a contiguous rectangle of black.
    for i in range(H):
        for j in range(W):
            if i < r_min or i > r_max or j < c_min or j > c_max:
                if grid[i][j] == '#':
                    print("No")
                    return

    print("Yes")

if __name__ == '__main__':
    main()