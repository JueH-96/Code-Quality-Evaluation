import sys

def solve():
    H_orig, W_orig = map(int, sys.stdin.readline().split())
    S_orig_str_list = [sys.stdin.readline().strip() for _ in range(H_orig)]

    MOD = 998244353

    # Determine effective H (num_dp_rows) and W (num_dp_cols/mask_width)
    # We want W (mask_width) to be the smaller dimension to minimize 3^W.
    # H (num_dp_rows) will be the larger dimension.
    
    H_dp_rows = H_orig
    W_dp_mask_width = W_orig
    S_processed_str_list = S_orig_str_list

    if W_orig > H_orig: # If original width > original height, transpose
        H_dp_rows = W_orig # Number of rows for DP is original width
        W_dp_mask_width = H_orig # Mask width for DP is original height (smaller)
        
        # Transpose S_orig_str_list
        # New grid S_processed_str_list will have W_orig rows and H_orig columns
        temp_S_list = [['' for _ in range(H_orig)] for _ in range(W_orig)]
        for r in range(H_orig):
            for c in range(W_orig):
                temp_S_list[c][r] = S_orig_str_list[r][c]
        S_processed_str_list = ["".join(row) for row in temp_S_list]

    # S_int_grid stores the grid with digits 0,1,2 and -1 for '?'
    S_int_grid = [[0] * W_dp_mask_width for _ in range(H_dp_rows)]
    for r in range(H_dp_rows):
        for c in range(W_dp_mask_width):
            char_val = S_processed_str_list[r][c]
            if char_val == '?':
                S_int_grid[r][c] = -1
            else:
                S_int_grid[r][c] = int(char_val) - 1 # Map '1','2','3' to 0,1,2

    # Precompute powers of 3
    POW3 = [1] * (W_dp_mask_width + 1)
    for i in range(1, W_dp_mask_width + 1):
        POW3[i] = POW3[i-1] * 3
    
    num_masks = POW3[W_dp_mask_width]
    dp_prev_row_values = [0] * num_masks

    # Base case: fill first row (row 0)
    for mask_idx in range(num_masks):
        current_config = [0] * W_dp_mask_width
        temp_mask = mask_idx
        is_valid_mask = True
        for c in range(W_dp_mask_width):
            digit = temp_mask % 3
            current_config[c] = digit
            temp_mask //= 3
            
            if S_int_grid[0][c] != -1 and S_int_grid[0][c] != current_config[c]:
                is_valid_mask = False
                break
            if c > 0 and current_config[c] == current_config[c-1]:
                is_valid_mask = False
                break
        
        if is_valid_mask:
            dp_prev_row_values[mask_idx] = 1

    # Iterate for remaining rows
    for r_dp in range(1, H_dp_rows):
        # Copy dp_prev_row_values to be transformed
        # This array will hold sums from previous row after FWT-like transform
        transformed_prev_row_sums = list(dp_prev_row_values) 

        # Apply FWT-like Transform
        for dim_idx in range(W_dp_mask_width): # For each dimension (column in the mask)
            p3_at_dim = POW3[dim_idx] # 3^dim_idx
            
            for i in range(num_masks):
                # Process each "fiber" once. A fiber consists of 3 masks
                # that differ only at dimension 'dim_idx'.
                # (i // p3_at_dim) % 3 == 0 means i is the smallest index in its fiber for this dim_idx.
                if (i // p3_at_dim) % 3 == 0:
                    idx0 = i
                    idx1 = i + p3_at_dim
                    idx2 = i + 2 * p3_at_dim
                    
                    val0 = transformed_prev_row_sums[idx0]
                    val1 = transformed_prev_row_sums[idx1]
                    val2 = transformed_prev_row_sums[idx2]
                    
                    transformed_prev_row_sums[idx0] = (val1 + val2) % MOD
                    transformed_prev_row_sums[idx1] = (val0 + val2) % MOD
                    transformed_prev_row_sums[idx2] = (val0 + val1) % MOD
        
        dp_current_row_values = [0] * num_masks
        for current_mask_idx in range(num_masks):
            if transformed_prev_row_sums[current_mask_idx] == 0:
                continue

            current_config = [0] * W_dp_mask_width
            temp_mask = current_mask_idx
            is_valid_mask = True
            for c in range(W_dp_mask_width):
                digit = temp_mask % 3
                current_config[c] = digit
                temp_mask //= 3

                if S_int_grid[r_dp][c] != -1 and S_int_grid[r_dp][c] != current_config[c]:
                    is_valid_mask = False
                    break
                if c > 0 and current_config[c] == current_config[c-1]:
                    is_valid_mask = False
                    break
            
            if is_valid_mask:
                dp_current_row_values[current_mask_idx] = transformed_prev_row_sums[current_mask_idx]
        
        dp_prev_row_values = dp_current_row_values

    total_ways = sum(dp_prev_row_values) % MOD
    print(total_ways)

solve()