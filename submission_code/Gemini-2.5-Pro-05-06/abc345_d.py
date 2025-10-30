import sys

def solve():
    N, H, W = map(int, sys.stdin.readline().split())
    tiles_orig = []
    for _ in range(N):
        tiles_orig.append(tuple(map(int, sys.stdin.readline().split())))

    # Iterate over all $2^N$ subsets of tiles
    for i in range(1 << N):
        current_subset_indices = []
        for j in range(N):
            if (i >> j) & 1: # If j-th bit is set, tile j is in subset
                current_subset_indices.append(j)
        
        k = len(current_subset_indices)

        if k == 0: # No tiles selected
            if H * W == 0: # If grid area is 0, empty set of tiles is a solution
                print("Yes")
                return
            else: # Grid area > 0, empty set is not a solution
                continue
        
        current_sum_area = 0
        subset_tiles_dims = [] 
        for tile_idx in current_subset_indices:
            A, B = tiles_orig[tile_idx]
            current_sum_area += A * B
            subset_tiles_dims.append(tiles_orig[tile_idx])
        
        if current_sum_area != H * W:
            continue

        board = [[False for _ in range(W)] for _ in range(H)]
        memo = {} 

        def backtrack_place(current_used_mask):
            r_empty, c_empty = -1, -1
            found_empty = False
            for r_idx in range(H):
                for c_idx in range(W):
                    if not board[r_idx][c_idx]:
                        r_empty, c_empty = r_idx, c_idx
                        found_empty = True
                        break
                if found_empty:
                    break
            
            if not found_empty:
                return True

            # Create hashable board representation for memoization key
            board_tuple_elements = []
            for row_iter_idx in range(H):
                board_tuple_elements.append(tuple(board[row_iter_idx]))
            board_tuple = tuple(board_tuple_elements)
            
            state = (board_tuple, current_used_mask)
            if state in memo:
                return memo[state]

            for j_idx_in_subset in range(k): # Iterate over tiles in the current subset
                if not ((current_used_mask >> j_idx_in_subset) & 1): # If this tile is not used yet
                    
                    tile_A, tile_B = subset_tiles_dims[j_idx_in_subset]
                    
                    orientations_to_try = [(tile_A, tile_B)]
                    if tile_A != tile_B:
                        orientations_to_try.append((tile_B, tile_A))
                    
                    for tile_h, tile_w in orientations_to_try:
                        if r_empty + tile_h <= H and c_empty + tile_w <= W:
                            can_be_placed_here = True
                            for row_to_fill in range(r_empty, r_empty + tile_h):
                                for col_to_fill in range(c_empty, c_empty + tile_w):
                                    if board[row_to_fill][col_to_fill]:
                                        can_be_placed_here = False
                                        break
                                if not can_be_placed_here:
                                    break
                            
                            if can_be_placed_here:
                                # Place the tile
                                for row_to_fill in range(r_empty, r_empty + tile_h):
                                    for col_to_fill in range(c_empty, c_empty + tile_w):
                                        board[row_to_fill][col_to_fill] = True
                                
                                new_mask = current_used_mask | (1 << j_idx_in_subset)
                                if backtrack_place(new_mask):
                                    memo[state] = True
                                    return True
                                
                                # Backtrack: Un-place the tile
                                for row_to_fill in range(r_empty, r_empty + tile_h):
                                    for col_to_fill in range(c_empty, c_empty + tile_w):
                                        board[row_to_fill][col_to_fill] = False
            
            memo[state] = False
            return False

        if backtrack_place(0): 
            print("Yes")
            return
            
    print("No")

solve()