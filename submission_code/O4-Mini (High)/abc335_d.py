def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    # Prepare empty grid
    grid = [[''] * N for _ in range(N)]
    # Generate the N^2 coordinates in a clockwise spiral
    positions = []
    top, bottom = 0, N - 1
    left, right = 0, N - 1
    while left <= right and top <= bottom:
        # move right along the top row
        for j in range(left, right + 1):
            positions.append((top, j))
        top += 1
        if top > bottom:
            break
        # move down along the right column
        for i in range(top, bottom + 1):
            positions.append((i, right))
        right -= 1
        if left > right:
            break
        # move left along the bottom row
        for j in range(right, left - 1, -1):
            positions.append((bottom, j))
        bottom -= 1
        if top > bottom:
            break
        # move up along the left column
        for i in range(bottom, top - 1, -1):
            positions.append((i, left))
        left += 1

    # The spiral visits all N^2 cells, the last one is the center when N is odd.
    total_cells = N * N
    # Fill the first N^2-1 cells in spiral order with 1..N^2-1
    for idx in range(total_cells - 1):
        i, j = positions[idx]
        grid[i][j] = str(idx + 1)
    # Place 'T' in the very last cell (the center)
    ci, cj = positions[-1]
    grid[ci][cj] = 'T'

    # Output the grid
    out = []
    for row in grid:
        out.append(' '.join(row))
    sys.stdout.write('
'.join(out))

if __name__ == "__main__":
    main()