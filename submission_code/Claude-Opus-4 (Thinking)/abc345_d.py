from itertools import permutations

def can_place_tiles_on_grid(tiles, H, W):
    # First check if the total area matches
    total_area = sum(a * b for a, b in tiles)
    if total_area != H * W:
        return False
    
    def can_place(tile_idx, tiles_order, orientations, grid):
        if tile_idx == len(tiles_order):
            return True
        
        tile = tiles_order[tile_idx]
        a, b = tile if orientations[tile_idx] == 0 else (tile[1], tile[0])
        
        # Try placing the tile at each position
        for i in range(H):
            for j in range(W):
                if i + a <= H and j + b <= W:
                    # Check if we can place the tile here
                    can_place_here = True
                    for di in range(a):
                        for dj in range(b):
                            if grid[i + di][j + dj]:
                                can_place_here = False
                                break
                        if not can_place_here:
                            break
                    
                    if can_place_here:
                        # Place the tile
                        for di in range(a):
                            for dj in range(b):
                                grid[i + di][j + dj] = True
                        
                        # Recurse
                        if can_place(tile_idx + 1, tiles_order, orientations, grid):
                            return True
                        
                        # Backtrack
                        for di in range(a):
                            for dj in range(b):
                                grid[i + di][j + dj] = False
        
        return False
    
    # Try all permutations and orientations
    for perm in permutations(tiles):
        # Try all orientations
        for orientations_mask in range(1 << len(tiles)):
            orientations = []
            for i in range(len(tiles)):
                orientations.append((orientations_mask >> i) & 1)
            
            # Reset grid
            grid = [[False] * W for _ in range(H)]
            
            if can_place(0, perm, orientations, grid):
                return True
    
    return False


# Read input
N, H, W = map(int, input().split())
all_tiles = []
for _ in range(N):
    a, b = map(int, input().split())
    all_tiles.append((a, b))

# Try all subsets of tiles
found = False
for mask in range(1, 1 << N):
    subset = []
    for i in range(N):
        if mask & (1 << i):
            subset.append(all_tiles[i])
    
    if can_place_tiles_on_grid(subset, H, W):
        found = True
        break

if found:
    print("Yes")
else:
    print("No")