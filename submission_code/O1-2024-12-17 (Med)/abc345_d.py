def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N, H, W = map(int, input_data[:3])
    tiles = list(zip(map(int, input_data[3::2]), map(int, input_data[4::2])))

    # Quick check: if total area of all tiles < area of grid, impossible
    total_area = sum(a*b for a,b in tiles)
    if total_area < H*W:
        print("No")
        return

    # covered[pos] indicates whether the cell at index pos is covered.
    # We'll flatten the grid so pos = r*W + c
    covered = [False] * (H*W)
    used = [False] * N  # whether a tile is used

    # Precompute each tile's area
    areas = [a*b for a,b in tiles]

    # For pruning: at any point, if uncovered_cells > sum(areas of unused tiles), we can prune
    def can_still_cover(uncovered_cells):
        # sum of areas of currently unused tiles
        return uncovered_cells <= sum(areas[i] for i in range(N) if not used[i])

    # Find row, col from flattened index
    def rc(pos):
        return divmod(pos, W)  # (row, col)

    # Check if we can place tile of size h x w with top-left corner at (r, c)
    def can_place(pos, h, w):
        r, c = rc(pos)
        if r + h > H or c + w > W:
            return False
        # Check if all needed cells are free
        for rr in range(r, r+h):
            base = rr * W
            for cc in range(c, c+w):
                if covered[base + cc]:
                    return False
        return True

    # Place (or remove) the tile, val=True to place, val=False to remove
    def set_covered(pos, h, w, val):
        r, c = rc(pos)
        for rr in range(r, r+h):
            base = rr * W
            for cc in range(c, c+w):
                covered[base + cc] = val

    def backtrack(pos, uncovered_cells):
        # If pos == H*W, we've checked all cells
        if pos == H*W:
            # If uncovered_cells == 0, all covered
            return (uncovered_cells == 0)

        # If this cell is already covered, skip to next
        if covered[pos]:
            return backtrack(pos+1, uncovered_cells)

        # If we don't have enough tile area left to cover all uncovered cells, prune
        if not can_still_cover(uncovered_cells):
            return False

        # Try each unused tile in each orientation
        r, c = rc(pos)
        for i in range(N):
            if not used[i]:
                a, b = tiles[i]
                # orientations
                for (hh, ww) in [(a,b), (b,a)]:
                    if can_place(pos, hh, ww):
                        # place the tile
                        set_covered(pos, hh, ww, True)
                        used[i] = True
                        if backtrack(pos+1, uncovered_cells - (hh*ww)):
                            return True
                        # revert
                        used[i] = False
                        set_covered(pos, hh, ww, False)

        # If no tile fits, fail
        return False

    # Start backtracking from pos = 0
    if backtrack(0, H*W):
        print("Yes")
    else:
        print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()