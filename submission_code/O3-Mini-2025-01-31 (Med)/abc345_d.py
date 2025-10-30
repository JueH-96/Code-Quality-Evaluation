def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    H = int(next(it))
    W = int(next(it))
    tiles = []
    for _ in range(N):
        a = int(next(it))
        b = int(next(it))
        tiles.append((a, b))
    total_area = H * W

    # We use a grid that will mark whether each cell is covered.
    grid = [[False] * W for _ in range(H)]
    
    # Define the backtracking recursion. 'remaining' is a list of tiles (each tile is a tuple (a,b))
    # that have not been placed yet.
    def backtrack(remaining):
        # Find the first empty cell (by scanning rows then columns).
        found = False
        for i in range(H):
            for j in range(W):
                if not grid[i][j]:
                    start_r, start_c = i, j
                    found = True
                    break
            if found:
                break
        # If no empty cell is found, then the grid is completely covered.
        if not found:
            return True
        
        # Try each remaining tile.
        for idx in range(len(remaining)):
            tile = remaining[idx]
            # Consider both orientations. Using a set avoids double-checking if tile is a square.
            orientations = {(tile[0], tile[1]), (tile[1], tile[0])}
            for (th, tw) in orientations:
                # Check boundary: the tile placed at (start_r, start_c) must not extend out.
                if start_r + th > H or start_c + tw > W:
                    continue
                # Check that all required cells are free.
                can_place = True
                for r in range(start_r, start_r + th):
                    for c in range(start_c, start_c + tw):
                        if grid[r][c]:
                            can_place = False
                            break
                    if not can_place:
                        break
                if not can_place:
                    continue
                # Place the tile by marking cells.
                for r in range(start_r, start_r + th):
                    for c in range(start_c, start_c + tw):
                        grid[r][c] = True
                # Prepare the remaining list, removing the tile that we just used.
                rem_next = remaining[:idx] + remaining[idx+1:]
                if backtrack(rem_next):
                    return True
                # Backtrack: remove the tile (unmark cells).
                for r in range(start_r, start_r + th):
                    for c in range(start_c, start_c + tw):
                        grid[r][c] = False
        return False
    
    # We can use only a subset of the available tiles. Thus, try every combination (subset)
    # of the tiles such that their total area exactly equals the grid area.
    from itertools import combinations
    possible = False
    for k in range(1, N+1):
        for subset in combinations(range(N), k):
            area_sum = 0
            candidate = []
            for idx in subset:
                a, b = tiles[idx]
                area_sum += a * b
                candidate.append((a, b))
            if area_sum != total_area:
                continue
            # Optionally, sort the candidate tiles in descending order by area,
            # which can sometimes help with pruning.
            candidate.sort(key=lambda x: x[0]*x[1], reverse=True)
            grid = [[False] * W for _ in range(H)]
            if backtrack(candidate):
                print("Yes")
                return
    print("No")

if __name__ == '__main__':
    main()