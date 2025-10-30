import sys

def solve():
    N, W = map(int, sys.stdin.readline().split())
    
    initial_blocks_data = [] 
    for i in range(1, N + 1):
        X, Y = map(int, sys.stdin.readline().split())
        initial_blocks_data.append({'X': X, 'Y': Y, 'id': i})

    # --- Simulate step t=1 ---

    # Check if bottom row is full based on initial coordinates
    initial_XY_set = set()
    for block in initial_blocks_data:
        initial_XY_set.add((block['X'], block['Y']))

    is_bottom_row_full_at_t1 = True
    if W == 0: # W >= 1 per constraints, but defensive.
        is_bottom_row_full_at_t1 = False
    else:
        for c_idx in range(1, W + 1):
            if (c_idx, 1) not in initial_XY_set:
                is_bottom_row_full_at_t1 = False
                break
    
    blocks_removed_at_t1_ids = set()
    if is_bottom_row_full_at_t1:
        for block in initial_blocks_data:
            if block['Y'] == 1:
                 blocks_removed_at_t1_ids.add(block['id'])

    surviving_blocks_for_fall = []
    for block in initial_blocks_data:
        if block['id'] not in blocks_removed_at_t1_ids:
            surviving_blocks_for_fall.append(block)

    # Perform falling for blocks in surviving_blocks_for_fall and assign ranks
    block_info_after_t1 = {} 
    blocks_by_col_for_ranking = [[] for _ in range(W + 1)]
    for block in surviving_blocks_for_fall:
        blocks_by_col_for_ranking[block['X']].append(block)

    counts_per_col_after_t1 = [0] * (W + 1) # Store k'_x values
    for c_idx in range(1, W + 1):
        blocks_by_col_for_ranking[c_idx].sort(key=lambda b: b['Y']) # Sort by original Y
        
        current_rank = 1
        for block in blocks_by_col_for_ranking[c_idx]:
            block_info_after_t1[block['id']] = {'rank': current_rank}
            current_rank += 1
        counts_per_col_after_t1[c_idx] = len(blocks_by_col_for_ranking[c_idx])

    # Calculate M_prime
    M_prime = float('inf')
    if W == 0:
        M_prime = 0
    else:
        any_column_empty_after_t1 = False
        for c_idx in range(1, W + 1):
            if counts_per_col_after_t1[c_idx] == 0:
                any_column_empty_after_t1 = True
                break
        
        if any_column_empty_after_t1:
            M_prime = 0
        else:
            # If not all columns became empty, find min height
            for c_idx in range(1, W + 1):
                 M_prime = min(M_prime, counts_per_col_after_t1[c_idx])
            if M_prime == float('inf'): # Only possible if W > 0 but no blocks at all (e.g. all removed, or N=0)
                 M_prime = 0
                 
    # --- Answer queries ---
    Q_count = int(sys.stdin.readline())
    results = []
    for _ in range(Q_count):
        T_j, A_j = map(int, sys.stdin.readline().split())

        if A_j in blocks_removed_at_t1_ids:
            results.append("No")
        else:
            # Block A_j survived t=1. Check its status based on further removals.
            # A_j must be in block_info_after_t1 if it was not removed at t=1.
            if A_j not in block_info_after_t1: # Should not happen for valid A_j
                results.append("No") # Safety check: if block somehow vanished
                continue

            rank_A_j_prime = block_info_after_t1[A_j]['rank']
            
            layers_removed_after_t1 = 0
            if T_j > 1: # Removals from t=2 onwards
                # max(0, T_j-1) is simply T_j-1 because T_j > 1 implies T_j-1 >= 1
                layers_removed_after_t1 = min(T_j - 1, M_prime) 
            
            if rank_A_j_prime > layers_removed_after_t1:
                results.append("Yes")
            else:
                results.append("No")
                
    sys.stdout.write("
".join(results) + "
")

solve()