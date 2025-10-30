def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N, H, W = map(int, input_data[:3])
    tiles = []
    idx = 3
    for _ in range(N):
        A, B = map(int, input_data[idx:idx+2])
        idx += 2
        tiles.append((A, B))
    
    # Total area needed
    total_area = H * W
    
    # Filter all subsets of tiles whose sum of areas == total_area
    from itertools import combinations, permutations
    
    valid_subsets = []
    for r in range(1, N+1):
        for combo in combinations(range(N), r):
            # sum the area of these tiles
            area_sum = sum(tiles[i][0] * tiles[i][1] for i in combo)
            if area_sum == total_area:
                valid_subsets.append(combo)
    
    # A backtracking function to see if we can place a given permutation of tiles
    # exactly covering the grid (H*W). We treat the grid as a 1D array of length H*W
    # Mark True if a cell is used, otherwise False.
    
    def place_tile(used, start, tileA, tileB):
        """ Try to place a tile of size tileA x tileB (height x width) 
            at linear index = start in row-major order.
            If successful, returns a list of updated cells that were marked.
            If fails, return None.
        """
        r = start // W
        c = start % W
        # Check bounds
        if r + tileA > H or c + tileB > W:
            return None
        
        cells_to_mark = []
        for rr in range(r, r + tileA):
            base_idx = rr * W
            for cc in range(c, c + tileB):
                lin = base_idx + cc
                if used[lin]:
                    return None
                cells_to_mark.append(lin)
        
        # Mark them
        for lin in cells_to_mark:
            used[lin] = True
        return cells_to_mark

    def unplace_tile(used, cells_list):
        """ Unmark the cells in cells_list. """
        for lin in cells_list:
            used[lin] = False

    def all_used(used):
        """ Check if all cells are used. """
        return all(used)
    
    def next_free(used, start):
        """ Return the next free index >= start in used array, or -1 if none. """
        for i in range(start, len(used)):
            if not used[i]:
                return i
        return -1
    
    def backtrack(used, tiles_perm, idx, start_free):
        """ 
        used: boolean array of length H*W 
        tiles_perm: list of (A, B) for the chosen permutation
        idx: index of the tile to place
        start_free: the linear index from where we search next free cell
        """
        if idx == len(tiles_perm):
            # If all tiles used, check coverage
            return all_used(used)
        
        # Find next free cell
        nf = next_free(used, start_free)
        if nf == -1:
            # no free cell
            # If we're out of free cells, check if all used
            return all_used(used)
        
        A, B = tiles_perm[idx]
        
        # Try orientation (A, B)
        placed = place_tile(used, nf, A, B)
        if placed is not None:
            if backtrack(used, tiles_perm, idx+1, nf):
                return True
            unplace_tile(used, placed)
        
        # If (A,B) != (B,A), try the flipped orientation
        if A != B:
            placed = place_tile(used, nf, B, A)
            if placed is not None:
                if backtrack(used, tiles_perm, idx+1, nf):
                    return True
                unplace_tile(used, placed)
        
        # If we cannot place the tile in any orientation at nf, fail
        return False

    # Try each valid subset in all permutations
    used_grid = [False]*(H*W)
    for subset in valid_subsets:
        subset_tiles = [tiles[i] for i in subset]
        # Try permutations of subset_tiles
        for perm in permutations(subset_tiles):
            # Clear the grid
            for i in range(H*W):
                used_grid[i] = False
            if backtrack(used_grid, perm, 0, 0):
                print("Yes")
                return
    
    # If no arrangement worked
    print("No")

# Do not forget to call the main function
if __name__ == "__main__":
    main()