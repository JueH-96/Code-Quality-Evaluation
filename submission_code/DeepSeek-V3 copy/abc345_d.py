# YOUR CODE HERE
import sys
from itertools import permutations

def can_fill_grid(N, H, W, tiles):
    total_area = H * W
    tile_areas = [a * b for a, b in tiles]
    # Check if the sum of tile areas is at least the total area
    if sum(tile_areas) < total_area:
        return False
    # Try all possible combinations of tiles
    # Since N is small (up to 7), we can afford to try all permutations
    for tile_perm in permutations(tiles):
        # Try to place the tiles in the grid
        grid = [[0 for _ in range(W)] for _ in range(H)]
        for tile in tile_perm:
            a, b = tile
            # Try both orientations
            for orientation in [(a, b), (b, a)]:
                a_curr, b_curr = orientation
                # Find a position to place the tile
                placed = False
                for i in range(H - a_curr + 1):
                    for j in range(W - b_curr + 1):
                        # Check if the area is available
                        available = True
                        for x in range(a_curr):
                            for y in range(b_curr):
                                if grid[i + x][j + y] != 0:
                                    available = False
                                    break
                            if not available:
                                break
                        if available:
                            # Place the tile
                            for x in range(a_curr):
                                for y in range(b_curr):
                                    grid[i + x][j + y] = 1
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
            else:
                # Could not place the tile in any orientation
                break
        else:
            # All tiles placed, check if the grid is fully covered
            if all(cell == 1 for row in grid for cell in row):
                return True
    return False

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    H = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    tiles = []
    for _ in range(N):
        A = int(input[idx])
        idx += 1
        B = int(input[idx])
        idx += 1
        tiles.append((A, B))
    if can_fill_grid(N, H, W, tiles):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()