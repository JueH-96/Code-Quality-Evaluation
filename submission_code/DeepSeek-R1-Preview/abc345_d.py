def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1

    tiles = []
    for _ in range(N):
        a = int(input[idx]); idx +=1
        b = int(input[idx]); idx +=1
        tiles.append((a, b))
    
    required_area = H * W

    # Function to check if the current subset of tiles can cover the grid
    def backtrack(tiles_remaining, grid):
        # Check if all cells are covered
        if all(all(row) for row in grid):
            return True
        
        # Find the first uncovered cell
        x, y = -1, -1
        for i in range(H):
            for j in range(W):
                if not grid[i][j]:
                    x, y = i, j
                    break
            if x != -1:
                break
        
        # Try all tiles in tiles_remaining
        for idx_tile in range(len(tiles_remaining)):
            tile = tiles_remaining[idx_tile]
            a, b = tile
            # Generate possible orientations
            orientations = []
            if a == b:
                orientations.append((a, b))
            else:
                orientations.append((a, b))
                orientations.append((b, a))
            
            for h, w in orientations:
                if x + h > H or y + w > W:
                    continue
                
                # Check if the tile can be placed without overlap
                can_place = True
                for i_place in range(x, x + h):
                    for j_place in range(y, y + w):
                        if grid[i_place][j_place]:
                            can_place = False
                            break
                    if not can_place:
                        break
                
                if can_place:
                    # Place the tile
                    for i_place in range(x, x + h):
                        for j_place in range(y, y + w):
                            grid[i_place][j_place] = True
                    
                    # Recurse with the remaining tiles
                    remaining = tiles_remaining[:idx_tile] + tiles_remaining[idx_tile+1:]
                    if backtrack(remaining, grid):
                        return True
                    
                    # Unplace the tile
                    for i_place in range(x, x + h):
                        for j_place in range(y, y + w):
                            grid[i_place][j_place] = False
        
        return False

    # Iterate over all possible subsets
    from itertools import combinations

    # We'll check all subsets via bitmask
    for bitmask in range(1, 1 << N):
        subset = []
        sum_area = 0
        for i in range(N):
            if (bitmask >> i) & 1:
                a, b = tiles[i]
                subset.append((a, b))
                sum_area += a * b
        
        if sum_area == required_area:
            # Create a new grid for this subset
            grid = [[False for _ in range(W)] for _ in range(H)]
            if backtrack(subset, grid):
                print("Yes")
                return
    
    print("No")

if __name__ == "__main__":
    main()