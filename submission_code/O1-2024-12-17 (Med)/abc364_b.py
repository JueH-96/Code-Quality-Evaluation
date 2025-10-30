def main():
    import sys

    data = sys.stdin.read().strip().split()
    H, W = map(int, data[:2])
    S_i, S_j = map(int, data[2:4])
    grid = data[4:4+H]
    X = data[4+H]

    # Current position (1-based indexing):
    r, c = S_i, S_j

    for move in X:
        if move == 'L':
            # Check if we can move left
            if c > 1 and grid[r - 1][c - 2] == '.':
                c -= 1
        elif move == 'R':
            # Check if we can move right
            if c < W and grid[r - 1][c] == '.':
                c += 1
        elif move == 'U':
            # Check if we can move up
            if r > 1 and grid[r - 2][c - 1] == '.':
                r -= 1
        elif move == 'D':
            # Check if we can move down
            if r < H and grid[r][c - 1] == '.':
                r += 1

    print(r, c)

# Do not remove, this is required to run the solution.
if __name__ == "__main__":
    main()