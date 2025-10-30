# YOUR CODE HERE
from itertools import permutations

def can_place_tiles(N, H, W, tiles):
    total_area = H * W
    tiles_area = [A * B for A, B in tiles]
    
    # Check if the total area of tiles is at least the area of the grid
    if sum(tiles_area) < total_area:
        return "No"
    
    # Generate all possible orientations of the tiles
    orientations = [(A, B), (B, A) for A, B in tiles]
    
    # Try all permutations of the tiles
    for perm in permutations(orientations):
        # Try placing the tiles in the grid
        grid = [[0] * W for _ in range(H)]
        placed = 0
        
        for A, B in perm:
            placed = False
            for i in range(H - A + 1):
                for j in range(W - B + 1):
                    if all(grid[x][y] == 0 for x in range(i, i + A) for y in range(j, j + B)):
                        for x in range(i, i + A):
                            for y in range(j, j + B):
                                grid[x][y] = 1
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                break
        
        if placed and all(grid[x][y] == 1 for x in range(H) for y in range(W)):
            return "Yes"
    
    return "No"

import sys
input = sys.stdin.read().split()
N = int(input[0])
H = int(input[1])
W = int(input[2])
tiles = [(int(input[3 + 2 * i]), int(input[4 + 2 * i])) for i in range(N)]

print(can_place_tiles(N, H, W, tiles))