import sys
from collections import defaultdict

def solve():
    N, W = map(int, sys.stdin.readline().split())

    blocks_by_col = defaultdict(list)
    # initial_block_positions is not strictly needed for the logic,
    # but storing original_idx is crucial.
    for i in range(1, N + 1):
        X_i, Y_i = map(int, sys.stdin.readline().split())
        blocks_by_col[X_i].append((Y_i, i))

    # Step 1 & 2: Sort blocks within each column
    for col in blocks_by_col:
        # Sorts by Y_i, then original_idx (original_idx tie-breaking is not critical here)
        blocks_by_col[col].sort() 

    # Step 3: Calculate initial_settled_Y and fall_time for each block
    # And populate max_fall_time_for_level for each effective Y-level (across W columns)
    block_initial_data = [None] * (N + 1) # stores (initial_settled_Y, fall_time)
    
    # max_fall_time_for_level[C] will store the maximum fall_time among
    # all blocks in the W columns that initially settle at level C.
    # Initialize with -1 to indicate no block has been found for that level yet,
    # or could be float('-inf') to correctly handle max().
    max_fall_time_for_level = defaultdict(lambda: -1) 
    
    # Track which columns provide blocks for each level
    cols_providing_level = defaultdict(set) # level -> set of column_ids

    # Iterate through columns 1 to W
    for x in range(1, W + 1):
        if x not in blocks_by_col:
            # If a required column (1 to W) is completely empty, 
            # no clears can ever happen across all W columns.
            # Mark all levels as impossible to clear.
            for c_val in range(1, N + 1): # max possible initial_settled_Y is N
                 max_fall_time_for_level[c_val] = float('inf')
            # No need to iterate further for other columns if one is empty.
            break 
        
        for rank, (Y_i, original_idx) in enumerate(blocks_by_col[x]):
            initial_settled_Y = rank + 1
            fall_time = Y_i - initial_settled_Y
            
            block_initial_data[original_idx] = (initial_settled_Y, fall_time)
            
            # Update the max fall time for this level C for blocks that are actually present
            # If max_fall_time_for_level[initial_settled_Y] is already infinity
            # (due to some other column not having this level), keep it infinity.
            if max_fall_time_for_level[initial_settled_Y] != float('inf'):
                max_fall_time_for_level[initial_settled_Y] = max(max_fall_time_for_level[initial_settled_Y], fall_time)
            
            cols_providing_level[initial_settled_Y].add(x)
            
        # After processing all blocks in column x, if this column does not have a block
        # for a certain level C, then mark that level C as impossible for this column.
        # This translates to setting its max_fall_time_for_level to infinity.
        for c_val in range(len(blocks_by_col[x]) + 1, N + 1):
            max_fall_time_for_level[c_val] = float('inf')

    # Final check: If any level C doesn't have a block from all W columns,
    # then that level cannot be cleared across all W columns.
    for c_val in range(1, N + 1):
        if len(cols_providing_level[c_val]) < W:
            max_fall_time_for_level[c_val] = float('inf')

    # Step 4: Calculate clearing_times
    # clearing_times[k-1] stores the actual time of the k-th clear.
    clearing_times = [] 
    
    # T_prev_clear_actual stores the actual time of the previous clear.
    # Initialized to 0, representing the state before any clears.
    T_prev_clear_actual = 0 

    for C_idx in range(1, N + 1): # Iterate through effective Y-levels (1-indexed)
        # time_last_block_lands_in_level_C is the max fall_time among blocks that
        # would form the C-th row when settled, across all W columns.
        time_last_block_lands_in_level_C = max_fall_time_for_level.get(C_idx, float('inf'))

        if time_last_block_lands_in_level_C == float('inf'):
            # This means not all W columns can provide a block for this level.
            # So, no further clears are possible.
            break

        # The actual time for the current clear (C_idx-th clear) is the maximum of:
        # 1. The time of the previous clear plus one time unit. (T_prev_clear_actual + 1)
        #    This accounts for the sequential nature of clearing and falling.
        # 2. The time the slowest block forming this C_idx-th effective row lands, plus one time unit.
        #    (+1 because removal happens at the next integer time step after blocks land and condition is met)
        current_clear_time = max(T_prev_clear_actual + 1, time_last_block_lands_in_level_C + 1)
        
        clearing_times.append(current_clear_time)
        T_prev_clear_actual = current_clear_time

    # Step 5: Process queries
    Q = int(sys.stdin.readline())
    results = []
    for _ in range(Q):
        T_j, A_j = map(int, sys.stdin.readline().split())

        initial_settled_Y_Aj, fall_time_Aj = block_initial_data[A_j]
        C_needed = initial_settled_Y_Aj # This is the effective Y-level (1-indexed) where block A_j would be removed

        if C_needed > len(clearing_times):
            # This means block A_j requires the C_needed-th clear to be removed,
            # but fewer than C_needed clears ever happen. So it is never removed.
            results.append("Yes")
        else:
            # time_removed is the actual time when the C_needed-th clear occurs.
            time_removed = clearing_times[C_needed - 1] # C_needed is 1-indexed, clearing_times is 0-indexed list
            
            # Query is for time T_j + 0.5.
            # If T_j < time_removed: block is removed after T_j + 0.5 (i.e., at time_removed or later). So it exists.
            # If T_j >= time_removed: block is removed at or before T_j + 0.5. So it does not exist.
            if T_j < time_removed:
                results.append("Yes")
            else:
                results.append("No")
    
    sys.stdout.write("
".join(results) + "
")

solve()