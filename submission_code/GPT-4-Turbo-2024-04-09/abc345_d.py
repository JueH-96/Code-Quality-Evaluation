def can_fill_grid(N, H, W, tiles):
    from itertools import permutations
    
    # Check if a set of tiles can fill the grid exactly
    def can_fill_using_tiles(permutation):
        grid = [[False] * W for _ in range(H)]
        
        # Try to place each tile in the permutation
        for tile in permutation:
            placed = False
            a, b = tile
            # Try to place the tile at every possible position
            for r in range(H):
                if placed:
                    break
                for c in range(W):
                    # Check if we can place the tile at position (r, c)
                    if all((r + i < H and c + j < W and not grid[r + i][c + j]) for i in range(a) for j in range(b)):
                        # Place the tile
                        for i in range(a):
                            for j in range(b):
                                grid[r + i][c + j] = True
                        placed = True
                        break
                    # Check if we can place the rotated tile at position (r, c)
                    if all((r + i < H and c + j < W and not grid[r + i][c + j]) for i in range(b) for j in range(a)):
                        # Place the rotated tile
                        for i in range(b):
                            for j in range(a):
                                grid[r + i][c + j] = True
                        placed = True
                        break
            if not placed:
                return False
        
        # Check if the entire grid is filled
        return all(all(row) for row in grid)
    
    # Generate all permutations of tiles and their rotations
    all_tiles = []
    for a, b in tiles:
        all_tiles.append((a, b))
        if a != b:
            all_tiles.append((b, a))
    
    # Try all permutations of tiles
    for perm in permutations(all_tiles):
        if can_fill_using_tiles(perm):
            return True
    
    return False

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
H = int(data[1])
W = int(data[2])
tiles = []

index = 3
for _ in range(N):
    A = int(data[index])
    B = int(data[index + 1])
    tiles.append((A, B))
    index += 2

# Check if it's possible to fill the grid
if can_fill_grid(N, H, W, tiles):
    print("Yes")
else:
    print("No")