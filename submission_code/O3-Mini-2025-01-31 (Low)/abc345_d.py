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
    tiles = []
    for i in range(N):
        a = int(next(it))
        b = int(next(it))
        tiles.append((a, b))
        
    total_area = H * W
    
    # We try to cover the grid exactly using a subset of these tiles.
    # Since N is small (<= 7), we consider every subset of tiles.
    subsets = []
    from itertools import combinations
    for subset_size in range(1, N+1):
        for indices in combinations(range(N), subset_size):
            area = 0
            for idx in indices:
                a, b = tiles[idx]
                area += a * b
            if area == total_area:
                subsets.append(indices)
    
    # For each candidate subset, try to tile the grid
    # We use backtracking over the grid: fill the first empty cell with any remaining tile (in either orientation)
    for subset in subsets:
        # selected_tiles is the list of candidate tiles (each used at most once)
        selected_tiles = [tiles[idx] for idx in subset]
        R = H
        C = W
        grid = [[False] * C for _ in range(R)]
        
        # Backtracking function: try to fill grid. used: list of booleans indicating if a tile from selected_tiles is used.
        def dfs(used):
            # Find the first empty cell in row-major order
            found = False
            for i in range(R):
                for j in range(C):
                    if not grid[i][j]:
                        si, sj = i, j
                        found = True
                        break
                if found:
                    break
            if not found:
                # grid is full
                return True
            
            # Attempt to place any unused tile in one of its orientations
            for t in range(len(selected_tiles)):
                if used[t]:
                    continue
                # Consider both orientations; if the tile is a square it is the same
                dims = []
                a, b = selected_tiles[t]
                dims.append((a, b))
                if a != b:
                    dims.append((b, a))
                for (height, width) in dims:
                    if si + height > R or sj + width > C:
                        continue
                    can_place = True
                    for di in range(height):
                        for dj in range(width):
                            if grid[si+di][sj+dj]:
                                can_place = False
                                break
                        if not can_place:
                            break
                    if not can_place:
                        continue
                    # Place tile
                    for di in range(height):
                        for dj in range(width):
                            grid[si+di][sj+dj] = True
                    used[t] = True
                    if dfs(used):
                        return True
                    # backtrack: remove tile
                    used[t] = False
                    for di in range(height):
                        for dj in range(width):
                            grid[si+di][sj+dj] = False
            return False
        
        used = [False] * len(selected_tiles)
        if dfs(used):
            print("Yes")
            return

    print("No")


if __name__ == '__main__':
    main()