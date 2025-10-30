def solve():
    N, H, W = map(int, input().split())
    tiles = []
    for _ in range(N):
        a, b = map(int, input().split())
        tiles.append((a, b))
    
    # Try all possible subsets of tiles
    for mask in range(1, 1 << N):
        selected_indices = []
        for i in range(N):
            if mask & (1 << i):
                selected_indices.append(i)
        
        # Try to place selected tiles with all orientation combinations
        if try_all_orientations(selected_indices, tiles, H, W, 0, []):
            print("Yes")
            return
    
    print("No")

def try_all_orientations(indices, tiles, H, W, pos, current_tiles):
    if pos == len(indices):
        return can_place_tiles(current_tiles, H, W)
    
    idx = indices[pos]
    a, b = tiles[idx]
    
    # Try original orientation
    current_tiles.append((a, b))
    if try_all_orientations(indices, tiles, H, W, pos + 1, current_tiles):
        return True
    current_tiles.pop()
    
    # Try rotated orientation (if different)
    if a != b:
        current_tiles.append((b, a))
        if try_all_orientations(indices, tiles, H, W, pos + 1, current_tiles):
            return True
        current_tiles.pop()
    
    return False

def can_place_tiles(tiles, H, W):
    grid = [[False] * W for _ in range(H)]
    return backtrack(tiles, 0, grid, H, W)

def backtrack(tiles, idx, grid, H, W):
    if idx == len(tiles):
        # Check if all cells are covered
        for i in range(H):
            for j in range(W):
                if not grid[i][j]:
                    return False
        return True
    
    tile_h, tile_w = tiles[idx]
    
    # Try placing current tile at each position
    for i in range(H - tile_h + 1):
        for j in range(W - tile_w + 1):
            # Check if we can place tile here
            can_place = True
            for di in range(tile_h):
                for dj in range(tile_w):
                    if grid[i + di][j + dj]:
                        can_place = False
                        break
                if not can_place:
                    break
            
            if can_place:
                # Place tile
                for di in range(tile_h):
                    for dj in range(tile_w):
                        grid[i + di][j + dj] = True
                
                # Try next tile
                if backtrack(tiles, idx + 1, grid, H, W):
                    return True
                
                # Backtrack
                for di in range(tile_h):
                    for dj in range(tile_w):
                        grid[i + di][j + dj] = False
    
    return False

solve()