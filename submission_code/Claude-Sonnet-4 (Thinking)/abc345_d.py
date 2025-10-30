N, H, W = map(int, input().split())
tiles = []
for _ in range(N):
    a, b = map(int, input().split())
    tiles.append((a, b))

# Grid to track which cells are covered
grid = [[False] * W for _ in range(H)]

def can_place(tile_h, tile_w, start_r, start_c):
    # Check if we can place a tile of size tile_h × tile_w at position (start_r, start_c)
    if start_r + tile_h > H or start_c + tile_w > W:
        return False
    
    for r in range(start_r, start_r + tile_h):
        for c in range(start_c, start_c + tile_w):
            if grid[r][c]:  # Cell already covered
                return False
    return True

def place_tile(tile_h, tile_w, start_r, start_c, place=True):
    # Place or remove a tile at position (start_r, start_c)
    for r in range(start_r, start_r + tile_h):
        for c in range(start_c, start_c + tile_w):
            grid[r][c] = place

def is_complete():
    # Check if all cells are covered
    for r in range(H):
        for c in range(W):
            if not grid[r][c]:
                return False
    return True

def backtrack(used_tiles):
    # Check if all cells are covered
    if is_complete():
        return True
    
    # Find next empty cell (top-left)
    for r in range(H):
        for c in range(W):
            if not grid[r][c]:
                # Try placing each unused tile at this position
                for i in range(N):
                    if used_tiles[i]:
                        continue
                    
                    a, b = tiles[i]
                    
                    # Try both orientations (avoid duplicate if a == b)
                    orientations = [(a, b)]
                    if a != b:
                        orientations.append((b, a))
                    
                    for tile_h, tile_w in orientations:
                        if can_place(tile_h, tile_w, r, c):
                            # Place the tile
                            place_tile(tile_h, tile_w, r, c, True)
                            used_tiles[i] = True
                            
                            if backtrack(used_tiles):
                                return True
                            
                            # Backtrack
                            place_tile(tile_h, tile_w, r, c, False)
                            used_tiles[i] = False
                
                # If no tile can be placed at this position, return False
                return False
    
    return False

used_tiles = [False] * N
if backtrack(used_tiles):
    print("Yes")
else:
    print("No")