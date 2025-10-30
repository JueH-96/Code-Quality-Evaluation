def main():
    import sys
    sys.setrecursionlimit(1000000)
    
    # Read input
    N, H, W = map(int, sys.stdin.readline().split())
    tiles = []
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        # Add both orientations if different
        if A != B:
            tiles.append([(A, B), (B, A)])
        else:
            tiles.append([(A, B)])
    
    # Precompute all possible tile orientations
    tile_orientations = []
    for tile in tiles:
        tile_orientations.append(tile)
    
    # Precompute grid area
    grid_area = H * W
    
    # Generate all subsets of tiles where sum of tile areas equals grid_area
    from itertools import combinations
    # Generate all possible subsets
    for subset_size in range(1, N+1):
        for subset in combinations(range(N), subset_size):
            # For each subset, generate all possible orientation combinations
            orientations = [tile_orientations[tile_idx] for tile_idx in subset]
            from itertools import product
            for orientation_comb in product(*orientations):
                total_area = sum(a * b for a, b in orientation_comb)
                if total_area != grid_area:
                    continue
                # Now, try to arrange these tiles on the grid
                grid = [[0 for _ in range(W)] for _ in range(H)]
                used = [False] * len(orientation_comb)
                # Define backtracking function
                def backtrack():
                    # Find the first empty cell
                    x, y = -1, -1
                    for i in range(H):
                        for j in range(W):
                            if grid[i][j] == 0:
                                x, y = i, j
                                break
                        if x != -1:
                            break
                    # If no empty cells, success
                    if x == -1:
                        return True
                    # Try placing each tile at this position
                    for t in range(len(orientation_comb)):
                        if not used[t]:
                            a, b = orientation_comb[t]
                            # Check if the tile fits starting at (x,y)
                            if x + a > H or y + b > W:
                                continue
                            # Check if the cells are empty
                            can_place = True
                            for dx in range(a):
                                for dy in range(b):
                                    if grid[x + dx][y + dy]:
                                        can_place = False
                                        break
                                if not can_place:
                                    break
                            if can_place:
                                # Mark the cells as occupied
                                for dx in range(a):
                                    for dy in range(b):
                                        grid[x + dx][y + dy] = 1
                                used[t] = True
                                # Recurse
                                if backtrack():
                                    return True
                                # Backtrack
                                for dx in range(a):
                                    for dy in range(b):
                                        grid[x + dx][y + dy] = 0
                                used[t] = False
                    return False
                # Try to place tiles starting from the first empty cell
                if backtrack():
                    print("Yes")
                    return
    # If no arrangement succeeded
    print("No")

if __name__ == "__main__":
    main()