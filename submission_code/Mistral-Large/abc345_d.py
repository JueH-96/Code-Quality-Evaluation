import sys
from itertools import combinations

def can_cover(grid, tiles):
    H, W = grid
    total_area = H * W
    tile_areas = [a * b for a, b in tiles]

    # Check if the total area of the tiles can cover the grid
    if sum(tile_areas) < total_area:
        return False

    # Generate all possible subsets of tiles
    for r in range(1, len(tiles) + 1):
        for subset in combinations(tiles, r):
            subset_areas = [a * b for a, b in subset]
            if sum(subset_areas) == total_area:
                # Check if the subset of tiles can be placed on the grid
                if can_place_tiles(grid, subset):
                    return True
    return False

def can_place_tiles(grid, tiles):
    H, W = grid
    grid_covered = [[False] * W for _ in range(H)]

    def place_tile(x, y, a, b):
        if x + a > H or y + b > W:
            return False
        for i in range(x, x + a):
            for j in range(y, y + b):
                if grid_covered[i][j]:
                    return False
        for i in range(x, x + a):
            for j in range(y, y + b):
                grid_covered[i][j] = True
        return True

    def backtrack(index):
        if index == len(tiles):
            return all(all(row) for row in grid_covered)
        a, b = tiles[index]
        for i in range(H):
            for j in range(W):
                if place_tile(i, j, a, b):
                    if backtrack(index + 1):
                        return True
                    # Remove the tile
                    for x in range(i, i + a):
                        for y in range(j, j + b):
                            grid_covered[x][y] = False
                if a != b:
                    if place_tile(i, j, b, a):
                        if backtrack(index + 1):
                            return True
                        # Remove the tile
                        for x in range(i, i + b):
                            for y in range(j, j + a):
                                grid_covered[x][y] = False
        return False

    return backtrack(0)

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    H = int(data[1])
    W = int(data[2])
    tiles = []
    for i in range(N):
        A = int(data[3 + 2 * i])
        B = int(data[4 + 2 * i])
        tiles.append((A, B))

    if can_cover((H, W), tiles):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()