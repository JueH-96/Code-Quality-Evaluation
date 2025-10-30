def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return

    # Parse input
    header = data[0].split()
    N = int(header[0])
    M = int(header[1])
    grid = data[1:]

    # Precompute the relative coordinates for cells that must be black and white.
    # For the 9x9 region, the top-left 3x3 block (relative positions 0-2,0-2) must be all black.
    black_top = [(r, c) for r in range(3) for c in range(3)]
    # The bottom-right 3x3 block (relative positions 6-8,6-8) must be all black.
    black_bottom = [(r, c) for r in range(6, 9) for c in range(6, 9)]

    # The white border adjacent to the top-left 3x3 block.
    # These are all the cells in the 4x4 block (rows 0-3 and cols 0-3)
    # minus the 3x3 block that is top-left.
    white_top = []
    for r in range(0, 4):
        for c in range(0, 4):
            if r < 3 and c < 3:
                continue  # skip the black 3x3
            white_top.append((r, c))

    # The white border adjacent to the bottom-right 3x3 block.
    # These are all the cells in the 4x4 block from rows 5-8 and cols 5-8
    # minus the bottom-right 3x3 block.
    white_bottom = []
    for r in range(5, 9):
        for c in range(5, 9):
            if r >= 6 and c >= 6:
                continue  # skip the black 3x3 block.
            white_bottom.append((r, c))

    results = []
    # Iterate over every candidate subgrid that is 9x9 and fully contained in the grid.
    for i in range(N - 9 + 1):
        for j in range(M - 9 + 1):
            isTakCode = True

            # Check the top-left 3x3 black region.
            for dr, dc in black_top:
                if grid[i + dr][j + dc] != '#':
                    isTakCode = False
                    break
            if not isTakCode:
                continue

            # Check the bottom-right 3x3 black region.
            for dr, dc in black_bottom:
                if grid[i + dr][j + dc] != '#':
                    isTakCode = False
                    break
            if not isTakCode:
                continue

            # Check the white border around the top-left block.
            for dr, dc in white_top:
                if grid[i + dr][j + dc] != '.':
                    isTakCode = False
                    break
            if not isTakCode:
                continue

            # Check the white border around the bottom-right block.
            for dr, dc in white_bottom:
                if grid[i + dr][j + dc] != '.':
                    isTakCode = False
                    break
            if not isTakCode:
                continue

            # If all conditions hold, store (i+1, j+1) as the result (converting to 1-indexed).
            results.append((i + 1, j + 1))

    # The candidates are already in lexicographical order by our iteration,
    # but we sort them to be sure.
    results.sort()

    # Output the results.
    for r, c in results:
        sys.stdout.write(f"{r} {c}
")
        
if __name__ == '__main__':
    main()