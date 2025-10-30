def can_place(tiles_indices, grid, H, W, index, tiles):
    if index == len(tiles_indices):
        return True
    current_tile_index = tiles_indices[index]
    a = tiles[current_tile_index][0]
    b = tiles[current_tile_index][1]
    
    # Try both orientations
    for h, w in [(a, b), (b, a)]:
        if h > H or w > W:
            continue
        
        # Try all possible positions
        for i in range(H - h + 1):
            for j in range(W - w + 1):
                # Check if all cells in this position are free
                valid = True
                for x in range(h):
                    for y in range(w):
                        if grid[i + x][j + y]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Mark the cells as occupied
                for x in range(h):
                    for y in range(w):
                        grid[i + x][j + y] = True
                
                # Recurse to place the next tile
                if can_place(tiles_indices, grid, H, W, index + 1, tiles):
                    return True
                
                # Unmark the cells (backtrack)
                for x in range(h):
                    for y in range(w):
                        grid[i + x][j + y] = False
    
    return False

# Read input
N, H, W = map(int, input().split())
tiles = []
for _ in range(N):
    A, B = map(int, input().split())
    tiles.append((A, B))

# Iterate over all possible subsets
for mask in range(1 << N):
    subset = []
    sum_area = 0
    for i in range(N):
        if (mask >> i) & 1:
            subset.append(i)
            sum_area += tiles[i][0] * tiles[i][1]
    
    if sum_area != H * W:
        continue
    
    # Check if all tiles in subset can fit in at least one orientation
    can_fit = True
    for i in subset:
        a = tiles[i][0]
        b = tiles[i][1]
        if (a <= H and b <= W) or (b <= H and a <= W):
            continue
        else:
            can_fit = False
            break
    if not can_fit:
        continue
    
    # Sort the subset in descending order of tile area to optimize backtracking
    subset_sorted = sorted(subset, key=lambda x: tiles[x][0] * tiles[x][1], reverse=True)
    
    # Create the grid and attempt to place the tiles
    grid = [[False for _ in range(W)] for _ in range(H)]
    if can_place(subset_sorted, grid, H, W, 0, tiles):
        print("Yes")
        exit()

# If no valid arrangement found
print("No")