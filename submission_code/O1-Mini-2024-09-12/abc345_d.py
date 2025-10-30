# YOUR CODE HERE
import sys
import itertools

def main():
    import sys
    sys.setrecursionlimit(10000)
    N, H, W = map(int, sys.stdin.readline().split())
    tiles = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        orientations = set()
        orientations.add( (a, b) )
        orientations.add( (b, a) )
        tiles.append(orientations)
    
    grid_area = H * W
    from collections import defaultdict

    # Precompute all subsets whose total area equals grid_area
    tile_areas = [len(o) for o in tiles]  # Not accurate, need to compute areas
    tile_areas = []
    for tile in tiles:
        areas = set()
        for (a,b) in tile:
            areas.add(a*b)
        tile_areas.append(areas)
    
    subsets = []
    for mask in range(1, 1<<N):
        area = 0
        for i in range(N):
            if mask & (1<<i):
                # For subset sum, need exact
                # But tiles can have multiple possible orientations but area is fixed
                # Take any area from tile_areas[i]
                # Since area is fixed for a tile regardless of orientation
                # So tiles have unique area
                area += next(iter(tile_areas[i]))
        if area == grid_area:
            subsets.append(mask)
    
    # Prepare tiles with their possible dimensions and areas
    tile_dims = []
    for tile in tiles:
        dims = list(tile)
        tile_dims.append(dims)
    
    from functools import lru_cache

    # Sort tiles in descending order of area for better pruning
    tile_info = []
    for i in range(N):
        a, b = next(iter(tile_dims[i]))
        tile_info.append( (a*b, i) )
    tile_info.sort(reverse=True)

    def can_place(grid, H, W, tiles_to_place):
        if not tiles_to_place:
            return True
        # Find first empty cell
        for i in range(H):
            for j in range(W):
                if grid[i][j] == False:
                    x, y = i, j
                    break
            else:
                continue
            break
        else:
            return True
        # Try to place each tile
        for idx, tile in enumerate(tiles_to_place):
            for (a, b) in tile_dims[tile]:
                if x + a > H or y + b > W:
                    continue
                can_fit = True
                for dx in range(a):
                    for dy in range(b):
                        if grid[x+dx][y+dy]:
                            can_fit = False
                            break
                    if not can_fit:
                        break
                if can_fit:
                    # Place the tile
                    for dx in range(a):
                        for dy in range(b):
                            grid[x+dx][y+dy] = True
                    # Recurse
                    if can_place(grid, H, W, tiles_to_place[:idx] + tiles_to_place[idx+1:]):
                        return True
                    # Remove the tile
                    for dx in range(a):
                        for dy in range(b):
                            grid[x+dx][y+dy] = False
        return False

    # Iterate over all subsets
    for mask in subsets:
        selected_tiles = []
        for i in range(N):
            if mask & (1<<i):
                selected_tiles.append(i)
        # Sort selected_tiles by descending area
        selected_tiles_sorted = sorted(selected_tiles, key=lambda x: -next(iter(tile_dims[x]))[0]*next(iter(tile_dims[x]))[1])
        grid = [[False]*W for _ in range(H)]
        if can_place(grid, H, W, selected_tiles_sorted):
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()