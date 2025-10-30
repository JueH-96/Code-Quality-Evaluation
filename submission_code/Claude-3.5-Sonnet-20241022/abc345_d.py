def can_cover(H, W, used, tiles, grid):
    if all(all(cell != 0 for cell in row) for row in grid):
        return True
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 0:
                for tile_idx in range(len(tiles)):
                    if used[tile_idx]:
                        continue
                        
                    tile = tiles[tile_idx]
                    # Try both orientations
                    for a, b in [(tile[0], tile[1]), (tile[1], tile[0])]:
                        if i + a > H or j + b > W:
                            continue
                            
                        # Check if we can place tile
                        can_place = True
                        for di in range(a):
                            for dj in range(b):
                                if grid[i+di][j+dj] != 0:
                                    can_place = False
                                    break
                            if not can_place:
                                break
                                
                        if can_place:
                            # Place tile
                            for di in range(a):
                                for dj in range(b):
                                    grid[i+di][j+dj] = tile_idx + 1
                            used[tile_idx] = True
                            
                            if can_cover(H, W, used, tiles, grid):
                                return True
                                
                            # Remove tile
                            for di in range(a):
                                for dj in range(b):
                                    grid[i+di][j+dj] = 0
                            used[tile_idx] = False
                            
                return False
    return False

N, H, W = map(int, input().split())
tiles = []
for _ in range(N):
    a, b = map(int, input().split())
    tiles.append((a, b))

grid = [[0]*W for _ in range(H)]
used = [False]*N

if can_cover(H, W, used, tiles, grid):
    print("Yes")
else:
    print("No")