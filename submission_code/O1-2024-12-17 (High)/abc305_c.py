def main():
    import sys
    data = sys.stdin.read().strip().split()
    H, W = map(int, data[:2])
    grid = data[2:]

    min_r, max_r = H, -1
    min_c, max_c = W, -1

    # Find the bounding rectangle of all '#' squares
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '#':
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # The missing cookie is the '.' inside that bounding rectangle
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == '.':
                print(r + 1, c + 1)
                return

# Do not forget to call main()
if __name__ == "__main__":
    main()