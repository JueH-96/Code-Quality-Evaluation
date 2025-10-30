def main():
    import sys
    sys.setrecursionlimit(10**7)
    N = int(sys.stdin.readline())
    
    # Generate the spiral order of all cells in an N x N grid
    spiral = []
    top, bottom = 0, N - 1
    left, right = 0, N - 1
    while top <= bottom and left <= right:
        # go right
        for c in range(left, right + 1):
            spiral.append((top, c))
        top += 1
        # go down
        for r in range(top, bottom + 1):
            spiral.append((r, right))
        right -= 1
        # go left
        if top <= bottom:
            for c in range(right, left - 1, -1):
                spiral.append((bottom, c))
            bottom -= 1
        # go up
        if left <= right:
            for r in range(bottom, top - 1, -1):
                spiral.append((r, left))
            left += 1

    # Prepare the grid
    grid = [[None] * N for _ in range(N)]
    center = (N // 2, N // 2)
    
    # Fill numbers 1 to N^2-1 along the spiral, leaving the last cell (center) for 'T'
    num = 1
    for (r, c) in spiral:
        if (r, c) == center:
            # skip numbering the center
            continue
        grid[r][c] = str(num)
        num += 1

    # Place Takahashi at the center
    grid[center[0]][center[1]] = 'T'

    # Output
    out = []
    for row in grid:
        out.append(" ".join(row))
    print("
".join(out))


if __name__ == "__main__":
    main()