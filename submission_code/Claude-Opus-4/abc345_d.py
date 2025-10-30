def can_place_tiles(H, W, tiles):
    # Grid to track covered cells
    grid = [[False] * W for _ in range(H)]
    
    def is_valid_placement(r, c, h, w):
        # Check if we can place a tile of size h×w at position (r, c)
        if r + h > H or c + w > W:
            return False
        for i in range(r, r + h):
            for j in range(c, c + w):
                if grid[i][j]:
                    return False
        return True
    
    def place_tile(r, c, h, w, value):
        # Place or remove a tile
        for i in range(r, r + h):
            for j in range(c, c + w):
                grid[i][j] = value
    
    def backtrack(tile_idx):
        # Check if all cells are covered
        all_covered = all(grid[i][j] for i in range(H) for j in range(W))
        
        if all_covered:
            return True
        
        if tile_idx >= len(tiles):
            return False
        
        # Find first uncovered cell
        start_r, start_c = -1, -1
        for i in range(H):
            for j in range(W):
                if not grid[i][j]:
                    start_r, start_c = i, j
                    break
            if start_r != -1:
                break
        
        if start_r == -1:  # All cells covered
            return True
        
        # Try placing current tile
        a, b = tiles[tile_idx]
        
        # Try both orientations
        for h, w in [(a, b), (b, a)]:
            if h == w and (a, b) == (b, a):  # Skip duplicate for square tiles
                if is_valid_placement(start_r, start_c, h, w):
                    place_tile(start_r, start_c, h, w, True)
                    if backtrack(tile_idx + 1):
                        return True
                    place_tile(start_r, start_c, h, w, False)
            else:
                if is_valid_placement(start_r, start_c, h, w):
                    place_tile(start_r, start_c, h, w, True)
                    if backtrack(tile_idx + 1):
                        return True
                    place_tile(start_r, start_c, h, w, False)
        
        # Try not using this tile
        if backtrack(tile_idx + 1):
            return True
        
        return False
    
    return backtrack(0)

# Read input
N, H, W = map(int, input().split())
tiles = []
for _ in range(N):
    a, b = map(int, input().split())
    tiles.append((a, b))

# Try all subsets of tiles
from itertools import combinations

for r in range(1, N + 1):
    for subset in combinations(range(N), r):
        selected_tiles = [tiles[i] for i in subset]
        # Check if total area matches
        total_area = sum(a * b for a, b in selected_tiles)
        if total_area == H * W:
            if can_place_tiles(H, W, selected_tiles):
                print("Yes")
                exit()

print("No")