from itertools import permutations

def can_place_tiles(N, H, W, tiles):
    def can_fill(grid, tile, r, c):
        for i in range(tile[0]):
            for j in range(tile[1]):
                if r + i >= H or c + j >= W or grid[r + i][c + j] != 0:
                    return False
        return True

    def place_tile(grid, tile, r, c):
        for i in range(tile[0]):
            for j in range(tile[1]):
                grid[r + i][c + j] = 1

    def remove_tile(grid, tile, r, c):
        for i in range(tile[0]):
            for j in range(tile[1]):
                grid[r + i][c + j] = 0

    def backtrack(grid, tiles, index):
        if index == len(tiles):
            return all(all(cell == 1 for cell in row) for row in grid)
        for r in range(H):
            for c in range(W):
                if can_fill(grid, tiles[index], r, c):
                    place_tile(grid, tiles[index], r, c)
                    if backtrack(grid, tiles, index + 1):
                        return True
                    remove_tile(grid, tiles[index], r, c)
        return False

    grid = [[0] * W for _ in range(H)]
    for perm in permutations(tiles):
        if backtrack(grid, perm, 0):
            return True
    return False

# Read input
N, H, W = map(int, input().split())
tiles = [tuple(map(int, input().split())) for _ in range(N)]

# Add all possible rotations/flips of the tiles
extended_tiles = []
for a, b in tiles:
    extended_tiles.append((a, b))
    if a != b:
        extended_tiles.append((b, a))

# Check if it's possible to place the tiles
if can_place_tiles(N, H, W, extended_tiles):
    print("Yes")
else:
    print("No")