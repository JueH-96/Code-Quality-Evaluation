def main():
    import sys
    input = sys.stdin.readline

    # Read input
    N, M = map(int, input().split())
    grid = [input().rstrip() for _ in range(N)]

    # We will iterate over every candidate 9x9 region.
    # For a candidate region with top-left cell at (i, j) (0-indexed), the region covers
    # rows i ... i+8 and columns j ... j+8.
    # The conditions for a valid Tak Code region are:
    #
    # 1. The top-left 3x3 block (cells with relative coordinates (0..2, 0..2)) must be black ('#').
    # 2. The bottom-right 3x3 block (cells with relative coordinates (6..8, 6..8)) must be black ('#').
    # 3. The white adjacent cells for the top-left block.
    #    The adjacent cells of a rectangular block are the cells that are next to it (touching it horizontally,
    #    vertically, or diagonally) and lie within the 9x9 region.
    #    For the top-left block (which covers rows 0..2 and columns 0..2), the adjacent cells (inside the 9x9 region)
    #    are those with relative positions:
    #         (0,3), (1,3), (2,3), (3,0), (3,1), (3,2), (3,3)
    #    All of these cells must be white ('.').
    # 4. Similarly, for the bottom-right block (covering rows 6..8 and columns 6..8), the adjacent cells within
    #    the 9x9 region are those with relative positions:
    #         (5,5), (5,6), (5,7), (5,8), (6,5), (7,5), (8,5)
    #    All of these must be white ('.').
    #
    # We check every candidate region and output in lexicographical order (i.e. increasing row then column)
    # using 1-indexing for the output.

    # Predefine the relative coordinates for the adjacent cells that must be white.
    top_adj = [(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    bottom_adj = [(5, 5), (5, 6), (5, 7), (5, 8), (6, 5), (7, 5), (8, 5)]

    out_lines = []
    # For each candidate 9x9 region
    for i in range(N - 8):
        for j in range(M - 8):
            valid = True
            # Check top-left 3x3 cells: they must be '#'
            for di in range(3):
                for dj in range(3):
                    if grid[i + di][j + dj] != '#':
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            # Check bottom-right 3x3 cells: they must be '#'
            for di in range(6, 9):
                for dj in range(6, 9):
                    if grid[i + di][j + dj] != '#':
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            # Check adjacent cells of the top-left 3x3 block: these 7 cells must be '.'
            for di, dj in top_adj:
                if grid[i + di][j + dj] != '.':
                    valid = False
                    break
            if not valid:
                continue

            # Check adjacent cells of the bottom-right 3x3 block: these 7 cells must be '.'
            for di, dj in bottom_adj:
                if grid[i + di][j + dj] != '.':
                    valid = False
                    break
            if not valid:
                continue

            # If all conditions are met, add the candidate's top-left position to the results.
            # (Output is 1-indexed.)
            out_lines.append(f"{i+1} {j+1}")
    
    # Output all valid top-left positions
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()