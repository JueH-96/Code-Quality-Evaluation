# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)

    N, H, W = map(int, sys.stdin.readline().split())
    tiles = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        tiles.append( (a,b) )

    grid = [[0]*W for _ in range(H)]

    tile_ids = list(range(1,N+1))  # Tile identifiers from 1 to N

    # For each tile, generate possible orientations
    tile_orientations = []
    for idx, (a,b) in enumerate(tiles):
        orientations = set()
        orientations.add((a,b))
        orientations.add((b,a))
        tile_orientations.append(list(orientations))

    # We need to keep track of which tiles are used
    used_tiles = [False]*N

    # The key is to proceed recursively, trying all possible placements

    # Function to check if we can place tile id at position (x, y) with size (h, w)
    def can_place(g, x, y, h, w):
        if x + h > H or y + w > W:
            return False
        for i in range(h):
            for j in range(w):
                if g[x + i][y + j] != 0:
                    return False
        return True

    # Place the tile id at position (x, y) with size (h, w)
    def place_tile(g, x, y, h, w, tile_id):
        for i in range(h):
            for j in range(w):
                g[x + i][y + j] = tile_id

    # Remove the tile id from position (x, y) with size (h, w)
    def remove_tile(g, x, y, h, w):
        for i in range(h):
            for j in range(w):
                g[x + i][y + j] = 0

    from itertools import permutations

    # For the tiles, generate all permutations of tile indices
    # Since N is small, we can consider all permutations
    tile_indices = list(range(N))
    # Let's try all subsets of tiles
    from itertools import combinations

    possible = False

    for k in range(1, N+1):
        subsets = combinations(tile_indices, k)
        for subset in subsets:
            # For each subset, we can generate permutations
            perms = permutations(subset)
            for perm in perms:
                used_tiles = [False]*N
                # reset the grid
                grid = [[0]*W for _ in range(H)]
                # Proceed to place tiles recursively
                def dfs():
                    # Find the next empty cell
                    for i in range(H):
                        for j in range(W):
                            if grid[i][j] == 0:
                                for idx in perm:
                                    if not used_tiles[idx]:
                                        for (h,w) in tile_orientations[idx]:
                                            if can_place(grid, i, j, h, w):
                                                used_tiles[idx] = True
                                                place_tile(grid, i, j, h, w, idx+1)
                                                if dfs():
                                                    return True
                                                remove_tile(grid, i, j, h, w)
                                                used_tiles[idx] = False
                                        # Try other orientation
                                # If no tile can be placed at this position, return False
                                return False
                    # If all cells are filled, return True
                    return True
                if dfs():
                    possible = True
                    break
            if possible:
                break
        if possible:
            break
    if possible:
        print("Yes")
        return

    # Also need to consider the case where we use unused tiles
    # Maybe we can start from empty grid and pick any tile at any position

    # Alternative approach: Proceed recursively, pick any unplaced tile, try all positions

    # Let's try a recursive function that tries to fill the grid

    from functools import lru_cache

    used_tiles = [False]*N
    grid = [[0]*W for _ in range(H)]
    def dfs():
        # Find the next empty cell
        for i in range(H):
            for j in range(W):
                if grid[i][j] == 0:
                    for idx in range(N):
                        if not used_tiles[idx]:
                            for (h,w) in tile_orientations[idx]:
                                for x in range(H - h +1):
                                    for y in range(W - w +1):
                                        # Check if we can place tile at (x,y)
                                        can_place_tile = True
                                        for dx in range(h):
                                            for dy in range(w):
                                                if grid[x + dx][y + dy] != 0:
                                                    can_place_tile = False
                                                    break
                                            if not can_place_tile:
                                                break
                                        if can_place_tile:
                                            used_tiles[idx] = True
                                            # Place the tile
                                            for dx in range(h):
                                                for dy in range(w):
                                                    grid[x + dx][y + dy] = idx +1
                                            if dfs():
                                                return True
                                            # Backtrack
                                            for dx in range(h):
                                                for dy in range(w):
                                                    grid[x + dx][y + dy] = 0
                                            used_tiles[idx] = False
                    # If we cannot place any tile at this position, return False
                    return False
        # If the grid is full
        # Check if all cells are filled
        for i in range(H):
            for j in range(W):
                if grid[i][j] == 0:
                    return False
        return True

    if dfs():
        print("Yes")
    else:
        print("No")


threading.Thread(target=main).start()