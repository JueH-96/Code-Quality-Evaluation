def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    # We create an N x N grid. Takahashi ("T") must go in the center.
    grid = [[None for _ in range(N)] for _ in range(N)]
    center = (N // 2, N // 2)
    grid[center[0]][center[1]] = "T"
    
    # We need to assign dragon parts 1...N^2-1 to every cell except the center.
    # Moreover, consecutive parts (x and x-1) must be placed in edge‐adjacent cells.
    # A constructive method is to produce a continuous Hamiltonian path that covers
    # all cells except the center. One standard way in an odd square grid is to “peel
    # off” layers in spiral order. In such a spiral ordering, the only leftover cell is
    # the center; and it turns out that the spiral ordering has the property that every
    # consecutive pair in the order is adjacent by an edge.
    #
    # We generate a spiral order (starting from the top‐left corner) and then remove
    # the center from that ordering.
    
    coords = []
    top, bottom = 0, N - 1
    left, right = 0, N - 1
    while left <= right and top <= bottom:
        # Traverse top row from left to right.
        for c in range(left, right + 1):
            coords.append((top, c))
        top += 1
        # Traverse right column from top to bottom.
        for r in range(top, bottom + 1):
            coords.append((r, right))
        right -= 1
        # Traverse bottom row from right to left (if still valid).
        if top <= bottom:
            for c in range(right, left - 1, -1):
                coords.append((bottom, c))
            bottom -= 1
        # Traverse left column from bottom to top (if still valid).
        if left <= right:
            for r in range(bottom, top - 1, -1):
                coords.append((r, left))
            left += 1

    # Remove the center cell from the ordering.
    coords = [pos for pos in coords if pos != center]
    
    # Now assign the dragon parts 1 to N^2-1 following the order in coords.
    # Since consecutive cells in the spiral ordering are adjacent, condition (3)
    # will be satisfied.
    part = 1
    for r, c in coords:
        grid[r][c] = str(part)
        part += 1

    # Output the grid.
    # Each row is printed with values separated by spaces.
    out_lines = []
    for row in grid:
        out_lines.append(" ".join(row))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()