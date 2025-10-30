def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    H = int(next(it))
    W = int(next(it))
    tiles = []
    for _ in range(N):
        a = int(next(it)); b = int(next(it))
        tiles.append((a, b))
    # Prepare areas and possible orientations
    areas = []
    shapes = []
    for a, b in tiles:
        areas.append(a * b)
        if a == b:
            shapes.append([(a, b)])
        else:
            shapes.append([(a, b), (b, a)])
    # Sort tiles by descending area to try larger tiles first
    idxs = list(range(N))
    idxs.sort(key=lambda i: areas[i], reverse=True)
    areas = [areas[i] for i in idxs]
    shapes = [shapes[i] for i in idxs]
    total_cells = H * W
    sum_areas = sum(areas)
    # Quick check: not enough total tile area
    if sum_areas < total_cells:
        print("No")
        return
    # Occupancy grid and used flags
    grid = [[False] * W for _ in range(H)]
    used = [False] * N

    def dfs(filled, rem_area):
        # If fully covered
        if filled == total_cells:
            return True
        # If even all remaining tiles can't fill the gap
        if rem_area < total_cells - filled:
            return False
        # Find the first empty cell (row-major)
        for r in range(H):
            for c in range(W):
                if not grid[r][c]:
                    r0, c0 = r, c
                    break
            else:
                continue
            break
        need = total_cells - filled
        # Try placing each unused tile at (r0, c0)
        for i in range(N):
            if used[i]:
                continue
            area_i = areas[i]
            # If this tile's area is too large for the remaining gap, skip
            if area_i > need:
                continue
            for h, w in shapes[i]:
                # Check bounds
                if r0 + h > H or c0 + w > W:
                    continue
                # Check overlap
                ok = True
                for rr in range(r0, r0 + h):
                    for cc in range(c0, c0 + w):
                        if grid[rr][cc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # Place tile i
                for rr in range(r0, r0 + h):
                    for cc in range(c0, c0 + w):
                        grid[rr][cc] = True
                used[i] = True
                # Recurse
                if dfs(filled + area_i, rem_area - area_i):
                    return True
                # Backtrack
                used[i] = False
                for rr in range(r0, r0 + h):
                    for cc in range(c0, c0 + w):
                        grid[rr][cc] = False
        return False

    ans = dfs(0, sum_areas)
    print("Yes" if ans else "No")

if __name__ == "__main__":
    main()