import sys
sys.setrecursionlimit(10000)

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N, H, W = map(int, (next(it), next(it), next(it)))
    A, B = [], []
    for _ in range(N):
        A.append(int(next(it)))
        B.append(int(next(it)))

    grid_area = H * W
    full_mask = (1 << grid_area) - 1
    tile_area = [A[i] * B[i] for i in range(N)]

    # ------------------------------------------------------------
    #  Pre–compute every legal placement (bit mask) for every tile
    # ------------------------------------------------------------
    placements_by_tile = [[] for _ in range(N)]
    for i in range(N):
        dims = {(A[i], B[i]), (B[i], A[i])}     # both orientations
        mask_set = set()
        for h, w in dims:
            if h > H or w > W:
                continue
            for r in range(H - h + 1):
                for c in range(W - w + 1):
                    m = 0
                    for dr in range(h):
                        row_start = (r + dr) * W + c
                        for dc in range(w):
                            m |= 1 << (row_start + dc)
                    mask_set.add(m)
        placements_by_tile[i] = list(mask_set)

    # quick impossibility: even using all tiles total area is too small
    if sum(tile_area) < grid_area:
        print("No")
        return

    # tiles ordered by descending area (helps pruning)
    order_desc = sorted(range(N), key=lambda idx: -tile_area[idx])

    from functools import lru_cache

    # -----------------------------------------------------------------
    #  Depth-first search that tries to tile using the chosen subset
    # -----------------------------------------------------------------
    @lru_cache(maxsize=None)
    def dfs(grid_mask: int, avail_mask: int) -> bool:
        if grid_mask == full_mask:                      # everything covered
            return True

        # find first uncovered cell (least significant free bit)
        free_bits = (~grid_mask) & full_mask
        anchor_bit = free_bits & -free_bits            # isolates LSB that is 1
        # iterate over still available tiles (larger first)
        for t in order_desc:
            if not (avail_mask & (1 << t)):
                continue
            for pm in placements_by_tile[t]:
                if (pm & anchor_bit) == 0:             # this placement doesn't hit anchor
                    continue
                if (pm & grid_mask):                   # overlaps something already placed
                    continue
                if dfs(grid_mask | pm, avail_mask ^ (1 << t)):
                    return True
            # no placement of tile t can cover anchor → try other tiles
        return False

    # ------------------------------------------------------------
    #  Enumerate every subset whose total area equals the grid area
    # ------------------------------------------------------------
    possible = False
    for submask in range(1, 1 << N):
        # area of this subset
        area_sum = sum(tile_area[i] for i in range(N) if (submask >> i) & 1)
        if area_sum != grid_area:
            continue
        # every tile in the subset must have at least one placement
        if any(not placements_by_tile[i] for i in range(N) if (submask >> i) & 1):
            continue
        # clear cache to keep it small between different subsets
        dfs.cache_clear()
        if dfs(0, submask):
            possible = True
            break

    print("Yes" if possible else "No")


if __name__ == "__main__":
    main()