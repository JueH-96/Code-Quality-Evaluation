# YOUR CODE HERE
import itertools

def can_cover_grid(N, H, W, tiles):
    # Generate all possible orientations of the tiles
    all_tiles = []
    for a, b in tiles:
        all_tiles.append((a, b))
        if a != b:
            all_tiles.append((b, a))
    
    # Check all permutations of the tiles
    for perm in itertools.permutations(all_tiles, N):
        if try_to_cover(H, W, perm):
            return True
    return False

def try_to_cover(H, W, tiles):
    grid = [[0] * W for _ in range(H)]
    
    def place_tile(x, y, h, w, tile_id):
        if x + h > H or y + w > W:
            return False
        for i in range(h):
            for j in range(w):
                if grid[x + i][y + j] != 0:
                    return False
        for i in range(h):
            for j in range(w):
                grid[x + i][y + j] = tile_id
        return True
    
    def remove_tile(x, y, h, w):
        for i in range(h):
            for j in range(w):
                grid[x + i][y + j] = 0
    
    def backtrack(tile_index):
        if tile_index == len(tiles):
            return all(all(cell != 0 for cell in row) for row in grid)
        
        tile_id = tile_index + 1
        h, w = tiles[tile_index]
        
        for x in range(H):
            for y in range(W):
                if place_tile(x, y, h, w, tile_id):
                    if backtrack(tile_index + 1):
                        return True
                    remove_tile(x, y, h, w)
        
        return False
    
    return backtrack(0)

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
H = int(data[1])
W = int(data[2])
tiles = [(int(data[2 * i + 3]), int(data[2 * i + 4])) for i in range(N)]

if can_cover_grid(N, H, W, tiles):
    print("Yes")
else:
    print("No")