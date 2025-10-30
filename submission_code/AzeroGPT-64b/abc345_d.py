from collections import Counter
from itertools import product

def solve(n, h, w, tiles):
    # check if there exists a valid covering combination for the grid
    def valid_combination(combination, total):
        for nbs in combination:
            if total - nbs == 0:
                return True
        return False

    # generate all possible orientations for each tile
    all_tiles = []
    for area, orientations in tiles.items():
        for orientation in orientations:
            all_tiles.append(orientation)
            all_tiles.append((orientation[1], orientation[0]))

    # check all possible combinations of tiles to cover the grid
    for tile_set in product(all_tiles, repeat=n):
        # calculate the number of times each tile size is used
        tile_counts = Counter(tile_set)
        area_counts = Counter()
        for tile, count in tile_counts.items():
            area_counts[tile[0] * tile[1]] += count

        # find if any valid combination of tile areas equals the total grid area
        if valid_combination([area * count for area, count in area_counts.items()], h * w):
            print("Yes")
            return

    # if no valid combination is found, it's not possible to tile the grid
    print("No")

# read input
n, h, w = map(int, input().split())
tiles = Counter()
for _ in range(n):
    a, b = map(int, input().split())
    tiles[(a, b)] += 1

solve(n, h, w, tiles)