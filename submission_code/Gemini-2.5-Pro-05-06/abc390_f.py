import sys
import bisect

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # positions[val] stores a sorted list of 0-indexed indices where val appears in A.
    # Values in A are 1 to N.
    # x_val ranges 1..N. x_val-1 ranges 0..N-1.
    # So, positions array needs to accommodate indices 0 through N. Size N+1.
    positions = [[] for _ in range(N + 1)]
    for i in range(N):
        positions[A[i]].append(i)

    total_f_sum = 0

    # Iterate over each possible value x_val that can start a "block"
    for x_val in range(1, N + 1):
        # If x_val is not in A, it cannot start a block for any S(L,R).
        if not positions[x_val]:
            continue

        # list_pos_x_minus_1 stores indices of (x_val-1)
        # If x_val is 1, x_val-1 is 0. positions[0] is an empty list, which is correct as 0 is not in A.
        list_pos_x_minus_1 = positions[x_val - 1]
        
        # Augmented list of positions of (x_val-1): add -1 and N as boundaries.
        # These define segments [segment_start_idx, segment_end_idx] where (x_val-1) does not appear.
        aug_list_pos_x_minus_1 = [-1] + list_pos_x_minus_1 + [N]

        current_x_val_total_contribution = 0

        # Iterate over segments defined by occurrences of (x_val-1)
        for k_segment in range(len(aug_list_pos_x_minus_1) - 1):
            # Current segment is A[segment_start_idx ... segment_end_idx]
            segment_start_idx = aug_list_pos_x_minus_1[k_segment] + 1
            segment_end_idx = aug_list_pos_x_minus_1[k_segment + 1] - 1

            if segment_start_idx > segment_end_idx: # Empty segment
                continue
            
            segment_len = segment_end_idx - segment_start_idx + 1
            
            # Total subarrays in A[segment_start_idx ... segment_end_idx]
            count_subarrays_in_segment = segment_len * (segment_len + 1) // 2
            
            # We need to subtract subarrays (within this segment) that DO NOT contain x_val.
            # These are subarrays fully within subsegments delimited by occurrences of x_val.

            # Find slice of positions[x_val] that falls within [segment_start_idx, segment_end_idx]
            idx_in_pos_list_for_start = bisect.bisect_left(positions[x_val], segment_start_idx)
            idx_in_pos_list_for_end = bisect.bisect_right(positions[x_val], segment_end_idx)

            current_scan_ptr = segment_start_idx - 1
            subarrays_to_subtract = 0 

            # Iterate over occurrences of x_val in the current segment A[segment_start_idx...segment_end_idx]
            for i_ptr in range(idx_in_pos_list_for_start, idx_in_pos_list_for_end):
                idx_of_x = positions[x_val][i_ptr] # Actual index in A where A[idx_of_x] == x_val
                
                # The subsegment A[current_scan_ptr + 1 ... idx_of_x - 1] does not contain x_val.
                len_subsegment_without_x = idx_of_x - (current_scan_ptr + 1)
                subarrays_to_subtract += len_subsegment_without_x * (len_subsegment_without_x + 1) // 2
                
                current_scan_ptr = idx_of_x # Update scan pointer
            
            # After iterating through all occurrences of x_val in the segment,
            # consider the final subsegment: A[current_scan_ptr + 1 ... segment_end_idx].
            # This subsegment also does not contain x_val.
            len_last_subsegment_without_x = segment_end_idx - (current_scan_ptr + 1) + 1
            subarrays_to_subtract += len_last_subsegment_without_x * (len_last_subsegment_without_x + 1) // 2
            
            contrib_from_this_segment = count_subarrays_in_segment - subarrays_to_subtract
            current_x_val_total_contribution += contrib_from_this_segment
            
        total_f_sum += current_x_val_total_contribution
        
    print(total_f_sum)

solve()