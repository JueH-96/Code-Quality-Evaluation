import sys
from collections import defaultdict

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    ops_params = []
    for _ in range(Q):
        p, v = map(int, sys.stdin.readline().split())
        ops_params.append((p, v))

    MOD = 998244353

    # dp maps state_tuple to count
    # Initial state: S is all 0s. Represented as ((0, N),).
    dp = {(((0, N),),): 1}

    for i_op_idx in range(Q):
        P_i, V_i = ops_params[i_op_idx]
        new_dp = defaultdict(int)

        for s_tuple, count in dp.items():
            # s_tuple is like ((val1, end_idx1), (val2, end_idx2), ...)
            # Segments are (value, 1-indexed_absolute_end_coordinate)
            # Example: [(10,3), (20,5)] means S[1..3]=10, S[4..5]=20 (for N=5)

            # --- Choice 1: Prefix S_1, ..., S_{P_i} with V_i ---
            max_val_prefix = 0
            current_scan_pos = 0 
            for k_val, k_end_idx in s_tuple:
                seg_start = current_scan_pos + 1
                
                overlap_start = seg_start 
                overlap_end = min(k_end_idx, P_i)

                if overlap_start <= overlap_end: # If there is an overlap with the checked prefix range [1, P_i]
                    max_val_prefix = max(max_val_prefix, k_val)
                
                current_scan_pos = k_end_idx
                if current_scan_pos >= P_i: 
                    break
            
            if max_val_prefix <= V_i:
                res_prefix_list = []
                # Add the new prefix segment
                if P_i > 0: # P_i is always >= 1 by constraints
                    res_prefix_list.append([V_i, P_i])
                
                current_scan_pos = 0
                for k_val, k_end_idx in s_tuple:
                    seg_start = current_scan_pos + 1
                    
                    # Consider the part of original segment (k_val, seg_start..k_end_idx)
                    # that is strictly after P_i
                    eff_seg_part_start = max(seg_start, P_i + 1)
                    eff_seg_part_end = k_end_idx 

                    if eff_seg_part_start <= eff_seg_part_end:
                        # This piece is (k_val, from eff_seg_part_start to eff_seg_part_end)
                        if res_prefix_list and res_prefix_list[-1][0] == k_val: # Check if mergeable with last segment
                            res_prefix_list[-1][1] = eff_seg_part_end 
                        else:
                            # This handles P_i=0 case if res_prefix_list is empty
                            # but P_i >= 1, so res_prefix_list has at least one element.
                            res_prefix_list.append([k_val, eff_seg_part_end])
                    
                    current_scan_pos = k_end_idx
                
                if not res_prefix_list and N > 0: # Should only occur if P_i=0 (not possible)
                                                 # or P_i=N and V_i=0 and original S was all 0s.
                                                 # If P_i=N, list is just [(V_i, N)]
                    res_prefix_list.append([0,N]) # Should be unreachable given N>=2, P_i>=1
                                                # Actually, if P_i=N, this list is just [(V_i,N)]
                                                # if V_i is 0, it's [(0,N)]
                
                new_s_tuple_prefix = tuple(map(tuple, res_prefix_list))
                new_dp[new_s_tuple_prefix] = (new_dp[new_s_tuple_prefix] + count) % MOD

            # --- Choice 2: Suffix S_{P_i}, ..., S_N with V_i ---
            max_val_suffix = 0
            current_scan_pos = 0
            for k_val, k_end_idx in s_tuple:
                seg_start = current_scan_pos + 1
                
                # Intersection with S_{P_i}...S_N
                overlap_start = max(seg_start, P_i)
                overlap_end = k_end_idx # min(k_end_idx, N) but k_end_idx is already <= N

                if overlap_start <= overlap_end:
                    max_val_suffix = max(max_val_suffix, k_val)
                
                current_scan_pos = k_end_idx
            
            if max_val_suffix <= V_i:
                res_suffix_list = []
                current_scan_pos = 0
                for k_val, k_end_idx in s_tuple:
                    seg_start = current_scan_pos + 1

                    # Part of this segment that is strictly before P_i
                    eff_seg_part_start = seg_start
                    eff_seg_part_end = min(k_end_idx, P_i - 1)

                    if eff_seg_part_start <= eff_seg_part_end:
                        if res_suffix_list and res_suffix_list[-1][0] == k_val:
                            res_suffix_list[-1][1] = eff_seg_part_end # Merge
                        else:
                            res_suffix_list.append([k_val, eff_seg_part_end])
                    
                    if k_end_idx >= P_i -1 : # This segment passed or touched P_i-1
                        break # No more segments fully before P_i
                    current_scan_pos = k_end_idx
                
                # Append the new suffix part S_{P_i}...S_N = V_i
                if res_suffix_list and res_suffix_list[-1][0] == V_i: # Merge with last prefix part
                    res_suffix_list[-1][1] = N 
                else: # Append as new segment
                    res_suffix_list.append([V_i, N])
                
                new_s_tuple_suffix = tuple(map(tuple, res_suffix_list))
                new_dp[new_s_tuple_suffix] = (new_dp[new_s_tuple_suffix] + count) % MOD
        
        dp = new_dp
        if not dp: # No valid sequence of operations can continue
            break

    total_ways = 0
    for count_val in dp.values():
        total_ways = (total_ways + count_val) % MOD
    
    print(total_ways)

solve()