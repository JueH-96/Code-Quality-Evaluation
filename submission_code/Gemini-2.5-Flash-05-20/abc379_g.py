import sys

MOD = 998244353

def solve():
    H, W = map(int, sys.stdin.readline().split())
    S_chars = [sys.stdin.readline().strip() for _ in range(H)]

    # Map '1','2','3' to 0,1,2 for base-3 representation
    # And transpose if H > W to ensure H is the smaller dimension
    original_H, original_W = H, W
    original_S_chars = S_chars

    S_mapped = [[0] * W for _ in range(H)] # Initialize with placeholder size

    if original_H > original_W:
        H, W = original_W, original_H
        S_mapped = [[0] * W for _ in range(H)]
        for r_orig in range(original_H):
            for c_orig in range(original_W):
                char = original_S_chars[r_orig][c_orig]
                val = -1 # Represents '?'
                if char != '?':
                    val = int(char) - 1 # Map 1->0, 2->1, 3->2
                S_mapped[c_orig][r_orig] = val
    else:
        S_mapped = [[0] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                char = original_S_chars[r][c]
                val = -1 # Represents '?'
                if char != '?':
                    val = int(char) - 1 # Map 1->0, 2->1, 3->2
                S_mapped[r][c] = val

    # Helper function to convert base-3 integer mask to a tuple of digits
    # mask_int: integer from 0 to 3^H - 1
    def int_to_base3_tuple(mask_int, dim_H):
        t = [0] * dim_H
        temp_mask = mask_int
        for i in range(dim_H):
            t[i] = temp_mask % 3
            temp_mask //= 3
        return tuple(t)

    # Pre-calculate all valid column masks (satisfying vertical constraints)
    # Store as a dictionary for O(1) lookup: integer mask -> tuple mask
    valid_column_masks_dict = {}
    
    # Iterate through all possible integer masks (3^H possibilities)
    # This loop runs 3^H times. Inside, a loop of H iterations.
    for mask_int in range(3**H):
        mask_tuple = int_to_base3_tuple(mask_int, H)
        is_vertically_valid = True
        for r in range(1, H):
            if mask_tuple[r] == mask_tuple[r-1]:
                is_vertically_valid = False
                break
        if is_vertically_valid:
            valid_column_masks_dict[mask_int] = mask_tuple

    # dp[mask_int] stores the number of ways to color columns up to j-1, with column j-1 having mask_int
    dp = {} 

    # Initialize dp for the first column (column 0)
    for mask_int in valid_column_masks_dict: # Iterate only over vertically valid masks
        mask_tuple = valid_column_masks_dict[mask_int] # O(1) lookup
        
        is_compatible_with_fixed = True
        for r in range(H):
            if S_mapped[r][0] != -1 and S_mapped[r][0] != mask_tuple[r]:
                is_compatible_with_fixed = False
                break
        
        if is_compatible_with_fixed:
            dp[mask_int] = 1
    
    # Iterate through columns from 1 to W-1
    for c in range(1, W):
        new_dp = {}
        for current_mask_int in valid_column_masks_dict: # Iterate only over vertically valid masks
            current_mask_tuple = valid_column_masks_dict[current_mask_int] # O(1) lookup
            
            # Check current_mask_tuple against fixed values in S_mapped[:,c]
            is_compatible_with_fixed = True
            for r in range(H):
                if S_mapped[r][c] != -1 and S_mapped[r][c] != current_mask_tuple[r]:
                    is_compatible_with_fixed = False
                    break
            
            if not is_compatible_with_fixed:
                continue

            # Sum up contributions from previous column's valid masks
            current_col_count = 0
            # Iterate only over masks present in dp (meaning they had valid ways to reach them)
            for prev_mask_int, prev_count in dp.items(): 
                prev_mask_tuple = valid_column_masks_dict[prev_mask_int] # O(1) lookup

                # Check horizontal constraints
                is_horizontal_compatible = True
                for r in range(H):
                    if current_mask_tuple[r] == prev_mask_tuple[r]:
                        is_horizontal_compatible = False
                        break
                
                if is_horizontal_compatible:
                    current_col_count = (current_col_count + prev_count) % MOD
            
            if current_col_count > 0:
                new_dp[current_mask_int] = current_col_count
        
        dp = new_dp # Move to next column
    
    # Sum up all counts in the final dp table
    total_count = 0
    for count in dp.values():
        total_count = (total_count + count) % MOD

    print(total_count)

solve()