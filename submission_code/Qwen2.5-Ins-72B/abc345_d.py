from itertools import permutations

def can_cover_grid(N, H, W, tiles):
    def can_place_tile(grid, tile, x, y):
        A, B = tile
        if x + A > H or y + B > W:
            return False
        for i in range(A):
            for j in range(B):
                if grid[x + i][y + j] != 0:
                    return False
        return True

    def place_tile(grid, tile, x, y):
        A, B = tile
        for i in range(A):
            for j in range(B):
                grid[x + i][y + j] = 1

    def remove_tile(grid, tile, x, y):
        A, B = tile
        for i in range(A):
            for j in range(B):
                grid[x + i][y + j] = 0

    def solve(grid, tiles, index):
        if index == len(tiles):
            for row in grid:
                if 0 in row:
                    return False
            return True

        for x in range(H):
            for y in range(W):
                for tile in [tiles[index], (tiles[index][1], tiles[index][0])]:
                    if can_place_tile(grid, tile, x, y):
                        place_tile(grid, tile, x, y)
                        if solve(grid, tiles, index + 1):
                            return True
                        remove_tile(grid, tile, x, y)
        return False

    for perm in permutations(tiles):
        grid = [[0] * W for _ in range(H)]
        if solve(grid, perm, 0):
            return "Yes"
    return "No"

N, H, W = map(int, input().split())
tiles = [tuple(map(int, input().split())) for _ in range(N)]
print(can_cover_grid(N, H, W, tiles))