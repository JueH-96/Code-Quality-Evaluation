def can_cover_grid(N, H, W, tiles):
    from itertools import permutations

    # Calculate the total area of the grid
    total_area = H * W

    # Generate all possible orientations of the tiles
    tile_orientations = []
    for a, b in tiles:
        tile_orientations.append([(a, b), (b, a)])  # Add both orientations

    # Check all permutations of the tiles
    for perm in permutations(tile_orientations):
        # Try to fill the grid with the current permutation of tiles
        for mask in range(1 << N):  # 2^N combinations of tiles
            current_area = 0
            used_tiles = []
            for i in range(N):
                if mask & (1 << i):  # If the i-th tile is used
                    used_tiles.append(perm[i])
                    current_area += perm[i][0][0] * perm[i][0][1]  # Add area of the first orientation
            if current_area == total_area:
                return "Yes"
            current_area = 0
            used_tiles = []
            for i in range(N):
                if mask & (1 << i):  # If the i-th tile is used
                    used_tiles.append(perm[i])
                    current_area += perm[i][1][0] * perm[i][1][1]  # Add area of the second orientation
            if current_area == total_area:
                return "Yes"

    return "No"

import sys
input = sys.stdin.read
data = input().splitlines()

N, H, W = map(int, data[0].split())
tiles = [tuple(map(int, line.split())) for line in data[1:N+1]]

print(can_cover_grid(N, H, W, tiles))