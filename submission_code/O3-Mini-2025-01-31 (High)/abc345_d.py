def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    H = int(next(it))
    W = int(next(it))
    
    # Read tiles; each tile can be used once.
    tiles = []
    areas = []
    for _ in range(N):
        a = int(next(it))
        b = int(next(it))
        tiles.append((a, b))
        areas.append(a * b)
    
    total_area = sum(areas)
    grid_area = H * W
    # Quick check: if even using all tiles we cannot cover the grid, it's impossible.
    if total_area < grid_area:
        sys.stdout.write("No")
        return
    
    # Precompute possible orientations for each tile.
    # Each orientation is a tuple (height, width). Rotation is allowed.
    tile_orients = []
    for (a, b) in tiles:
        if a == b:
            tile_orients.append([(a, b)])
        else:
            tile_orients.append([(a, b), (b, a)])
    
    # We'll represent the grid as a list of integers.
    # Each integer is a bitmask for a row, with W bits: if bit j is set, then cell (r,j) is covered.
    grid = [0] * H
    FULL = (1 << W) - 1  # mask when a row is completely filled.
    
    # Memo dictionary to store states we have already processed.
    memo = {}
    
    def count_free(grid):
        free = 0
        for r in range(H):
            # Count number of 0 bits in grid[r] for width W.
            # Since W is small, we can simply do:
            free += W - bin(grid[r]).count("1")
        return free
    
    def dfs(used_mask, grid):
        state = (tuple(grid), used_mask)
        if state in memo:
            return False
        # Compute remaining free cells.
        free = count_free(grid)
        # Compute sum of areas available from unused tiles.
        available_area = 0
        for i in range(N):
            if (used_mask >> i) & 1 == 0:
                available_area += areas[i]
        # Prune: if there is not enough area left in unused tiles.
        if available_area < free:
            memo[state] = False
            return False

        # Find first empty cell in row-major order.
        found = False
        for r in range(H):
            if grid[r] != FULL:
                # find the first column in this row that is free.
                for c in range(W):
                    if ((grid[r] >> c) & 1) == 0:
                        r0, c0 = r, c
                        found = True
                        break
                if found:
                    break
        # If no free cell found, grid is completely covered.
        if not found:
            return True

        # Try to cover cell (r0, c0) with one of the available tiles.
        for i in range(N):
            if (used_mask >> i) & 1:
                continue  # already used tile i.
            # Try each orientation.
            for (th, tw) in tile_orients[i]:
                # If the tile is larger than grid dimensions, skip.
                if th > H or tw > W:
                    continue
                # We need to place the tile so that cell (r0, c0) is inside the tile placement.
                # Let the top-left corner of the tile be at (r, c).
                # Then we require: r <= r0 < r + th and c <= c0 < c + tw.
                # Also, the tile must fit in the grid:
                #    r in [0, H - th] and c in [0, W - tw].
                # So r can range from max(0, r0 - th + 1) to min(r0, H - th)
                rmin = max(0, r0 - th + 1)
                rmax = min(r0, H - th)
                # Similarly, c can range from max(0, c0 - tw + 1) to min(c0, W - tw).
                cmin = max(0, c0 - tw + 1)
                cmax = min(c0, W - tw)
                for rr in range(rmin, rmax + 1):
                    for cc in range(cmin, cmax + 1):
                        # Check if the rectangle from (rr, cc) with size (th x tw) is free.
                        can_place = True
                        mask = ((1 << tw) - 1) << cc  # mask for columns cc .. cc+tw-1.
                        for rTile in range(rr, rr + th):
                            if grid[rTile] & mask:
                                can_place = False
                                break
                        if not can_place:
                            continue
                        # Place the tile: mark the cells in the rectangle.
                        for rTile in range(rr, rr + th):
                            grid[rTile] |= mask
                        new_used_mask = used_mask | (1 << i)
                        if dfs(new_used_mask, grid):
                            return True
                        # Backtrack: unmark the cells.
                        for rTile in range(rr, rr + th):
                            grid[rTile] ^= mask
        memo[state] = False
        return False

    result = dfs(0, grid)
    sys.stdout.write("Yes" if result else "No")

if __name__ == '__main__':
    main()