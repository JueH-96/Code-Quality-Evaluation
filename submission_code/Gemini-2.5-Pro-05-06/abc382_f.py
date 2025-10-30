import bisect
from collections import defaultdict

def solve():
    H, W, N_bars = map(int, input().split())
    
    # Store bar properties. 'id' is the original 0-indexed position in input.
    bars_props = []
    for i in range(N_bars):
        r, c, l = map(int, input().split())
        # Store 'id': i, which is its 0-indexed original ID.
        bars_props.append({'r': r, 'c': c, 'l': l, 'id': i})

    # current_R[bar_id] stores current row of bar with 'id'
    # Initialize with input rows
    current_R = [bar['r'] for bar in bars_props]
    
    # row_to_intervals[row_num] = sorted list of (c_start, c_end, bar_id)
    # bar_id is the 'id' field (0 to N_bars-1)
    row_to_intervals = defaultdict(list)

    for i in range(N_bars):
        bar = bars_props[i]
        r, c, l, bar_id = bar['r'], bar['c'], bar['l'], bar['id']
        
        interval = (c, c + l - 1, bar_id) # c_end is inclusive
        
        # Insert into sorted list for that row
        list_for_row = row_to_intervals[r]
        # Insort_left maintains sort order and is consistent with bisect_left for lookups
        bisect.insort_left(list_for_row, interval)

    while True:
        moved_in_pass = False
        # Process bars in order of their original index 0 to N_bars-1
        for bar_id_to_process in range(N_bars):
            
            r_k = current_R[bar_id_to_process]
            # Properties of bar_k (c_k, l_k) are fixed
            c_k = bars_props[bar_id_to_process]['c']
            l_k = bars_props[bar_id_to_process]['l']
            
            cols_start_k = c_k
            cols_end_k = c_k + l_k - 1
            
            # If bar is on the bottom row, it cannot move
            if r_k == H:
                continue

            target_R_k = r_k + 1
            is_blocked = False
            
            # Check if any bar in target_R_k blocks the current bar
            # Only access row_to_intervals[target_R_k] if it exists (i.e., target_R_k in row_to_intervals)
            if target_R_k in row_to_intervals:
                intervals_in_target_row = row_to_intervals[target_R_k]
                # intervals_in_target_row could be empty if all bars moved out of it.
                if intervals_in_target_row: 
                    # Query interval for bar k: [cols_start_k, cols_end_k]
                    
                    # Find idx such that all e in intervals_in_target_row[:idx] have e[0] < cols_start_k,
                    # and all e in intervals_in_target_row[idx:] have e[0] >= cols_start_k.
                    # Intervals are (c_start, c_end, bar_id). We search based on c_start.
                    # Using float('-inf') for other tuple elements to ensure primary sort key dominance.
                    idx = bisect.bisect_left(intervals_in_target_row, (cols_start_k, float('-inf'), float('-inf')))
                    
                    # Check interval at intervals_in_target_row[idx] (if exists)
                    # This interval (s_other, e_other, _) has s_other >= cols_start_k.
                    # It overlaps if s_other <= cols_end_k.
                    if idx < len(intervals_in_target_row):
                        s_other, _, _ = intervals_in_target_row[idx] # e_other not needed for this condition
                        if s_other <= cols_end_k:
                            is_blocked = True
                    
                    # Check interval at intervals_in_target_row[idx-1] (if exists)
                    # This interval (s_other, e_other, _) has s_other < cols_start_k.
                    # It overlaps if e_other >= cols_start_k.
                    if not is_blocked and idx > 0:
                        _, e_other, _ = intervals_in_target_row[idx-1] # s_other not needed for this condition
                        if e_other >= cols_start_k:
                            is_blocked = True
            
            if not is_blocked:
                # Move bar: remove from old row, add to new row
                interval_to_remove = (cols_start_k, cols_end_k, bar_id_to_process)
                
                old_row_list = row_to_intervals[r_k]
                # Find the exact interval to remove. bisect_left gives its index.
                remove_idx = bisect.bisect_left(old_row_list, interval_to_remove)
                old_row_list.pop(remove_idx) # O(K) operation
                
                if not old_row_list: # If row becomes empty, remove key from dict
                    del row_to_intervals[r_k]
                
                current_R[bar_id_to_process] = target_R_k
                
                interval_to_add = (cols_start_k, cols_end_k, bar_id_to_process)
                new_row_list = row_to_intervals[target_R_k] # defaultdict creates if not exist
                bisect.insort_left(new_row_list, interval_to_add) # O(K) operation
                
                moved_in_pass = True
        
        if not moved_in_pass: # If no bar moved in a full pass, system is stable
            break
            
    # Output final rows
    for r_final in current_R:
        print(r_final)

solve()