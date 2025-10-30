def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    H = int(next(it))
    W = int(next(it))
    tiles = []
    total_area = 0
    for _ in range(N):
        a = int(next(it)); b = int(next(it))
        area = a * b
        tiles.append([a, b, area])
        total_area += area

    # If even all tiles area < grid area, impossible
    if total_area < H * W:
        print("No")
        return

    # Sort tiles by descending area to prune earlier
    tiles.sort(key=lambda x: x[2], reverse=True)

    grid = [[False]*W for _ in range(H)]
    used = [False]*N

    # Backtracking
    def dfs(rem_empty, rem_tile_area):
        # If no empty cells, success
        if rem_empty == 0:
            return True
        # If not enough tile area to fill, fail
        if rem_tile_area < rem_empty:
            return False

        # Find first empty cell in row-major order
        for r in range(H):
            for c in range(W):
                if not grid[r][c]:
                    start_r, start_c = r, c
                    r = H  # break outer
                    break
        else:
            # no empty found
            return True

        # Try placing each unused tile at (start_r, start_c)
        for i in range(N):
            if used[i]:
                continue
            a, b, area = tiles[i]
            # Try both orientations
            for h, w in ((a, b), (b, a)):
                # skip duplicate when square
                if a == b and h != a:
                    continue
                if start_r + h > H or start_c + w > W:
                    continue
                # Check if placement overlaps
                ok = True
                for rr in range(start_r, start_r + h):
                    for cc in range(start_c, start_c + w):
                        if grid[rr][cc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # Place tile
                for rr in range(start_r, start_r + h):
                    for cc in range(start_c, start_c + w):
                        grid[rr][cc] = True
                used[i] = True

                if dfs(rem_empty - area, rem_tile_area - area):
                    return True

                # Unplace tile
                used[i] = False
                for rr in range(start_r, start_r + h):
                    for cc in range(start_c, start_c + w):
                        grid[rr][cc] = False
            # end orientations
        # end tiles loop

        return False

    ans = dfs(H*W, total_area)
    print("Yes" if ans else "No")

if __name__ == "__main__":
    main()