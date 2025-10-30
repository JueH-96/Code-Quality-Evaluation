import sys
sys.setrecursionlimit(10000)

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    H = int(data[1])
    W = int(data[2])
    tiles = []
    idx = 3
    for _ in range(N):
        a = int(data[idx]); b = int(data[idx+1])
        idx += 2
        tiles.append((a, b))

    # Total area needed
    total_area = H * W
    # Precompute each tile's area
    areas = [a*b for (a, b) in tiles]
    # If sum of all tile areas < grid area, impossible
    if sum(areas) < total_area:
        print("No")
        return

    # Grid to mark coverage
    grid = [[False]*W for _ in range(H)]
    # Bitmask to mark used tiles
    # We will also precompute sum of areas for masks for pruning
    sum_area_for_mask = {}
    def compute_sum(mask):
        s = 0
        for i in range(N):
            if (mask>>i)&1:
                s += areas[i]
        return s
    # Precompute for all masks 0..(1<<N)-1
    for m in range(1<<N):
        sum_area_for_mask[m] = compute_sum(m)

    # Find next empty cell (row, col), or None if full
    def find_empty():
        for r in range(H):
            for c in range(W):
                if not grid[r][c]:
                    return (r, c)
        return None

    # Check if we can place a tile of size h x w at (r,c)
    def can_place(r, c, h, w):
        if r + h > H or c + w > W:
            return False
        for i in range(r, r+h):
            for j in range(c, c+w):
                if grid[i][j]:
                    return False
        return True

    # Place or remove a tile (fill=True to place, False to unplace)
    def set_place(r, c, h, w, fill):
        for i in range(r, r+h):
            for j in range(c, c+w):
                grid[i][j] = fill

    # DFS backtracking
    from functools import lru_cache

    @lru_cache(None)
    def dfs(used_mask):
        # Prune: if area of used tiles < total_area => check remaining coverage
        used_area = sum_area_for_mask[used_mask]
        # remaining empty must be total_area - covered = total_area - used_area
        if used_area > total_area:
            return False
        # Find first empty cell
        emp = find_empty()
        if emp is None:
            # No empty => fully covered
            return used_area == total_area
        r0, c0 = emp
        # Try to place any unused tile here
        for i in range(N):
            if (used_mask>>i)&1 == 0:
                a, b = tiles[i]
                for (h, w) in ((a, b), (b, a)):
                    if can_place(r0, c0, h, w):
                        # place
                        set_place(r0, c0, h, w, True)
                        if dfs(used_mask | (1<<i)):
                            return True
                        # undo
                        set_place(r0, c0, h, w, False)
        return False

    ans = dfs(0)
    print("Yes" if ans else "No")

if __name__ == "__main__":
    main()