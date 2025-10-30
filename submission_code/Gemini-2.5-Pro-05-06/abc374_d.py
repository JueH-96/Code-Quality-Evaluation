import math

def solve():
    N, S_speed, T_speed = map(int, input().split())
    
    # Store segment endpoints. P_i0 is (A_i, B_i), P_i1 is (C_i, D_i).
    # Using floats for coordinates from the start for consistency in calculations.
    segments_coords = []
    for _ in range(N):
        A, B, C, D = map(int, input().split())
        segments_coords.append(((float(A), float(B)), (float(C), float(D))))

    # Precompute sum of (Length_i / T_speed) for all segments. This is a fixed part of total time.
    total_printing_time_fixed_part = 0.0
    for i in range(N):
        P_i0 = segments_coords[i][0] 
        P_i1 = segments_coords[i][1] 
        
        length = math.hypot(P_i0[0] - P_i1[0], P_i0[1] - P_i1[1])
        total_printing_time_fixed_part += length / float(T_speed)

    # dp[mask][last_idx][end_type] stores minimum travel time (moving without printing).
    # end_type=0: laser ended at P_last_idx,0
    # end_type=1: laser ended at P_last_idx,1
    dp = [[[float('inf')] * 2 for _ in range(N)] for _ in range(1 << N)]

    origin = (0.0, 0.0) # Laser starts at (0,0)

    # Base cases: for printing the first segment
    for i in range(N):
        mask = 1 << i # Mask representing only segment i is printed
        
        P_i0 = segments_coords[i][0]
        P_i1 = segments_coords[i][1]
        
        # Option 1: Print P_i1 -> P_i0. Laser ends at P_i0 (end_type=0).
        # Laser must move from origin to P_i1 to start printing.
        dist_to_P_i1 = math.hypot(origin[0] - P_i1[0], origin[1] - P_i1[1])
        dp[mask][i][0] = dist_to_P_i1 / float(S_speed)
        
        # Option 2: Print P_i0 -> P_i1. Laser ends at P_i1 (end_type=1).
        # Laser must move from origin to P_i0 to start printing.
        dist_to_P_i0 = math.hypot(origin[0] - P_i0[0], origin[1] - P_i0[1])
        dp[mask][i][1] = dist_to_P_i0 / float(S_speed)

    # Fill DP table
    # Iterate mask from 1 up to (1<<N)-1. Masks with more set bits are processed later.
    # This order ensures that dp[prev_mask] is computed before dp[mask].
    for mask in range(1, 1 << N):
        for curr_i in range(N):
            # curr_i is the segment just printed to reach this state (mask)
            if not ((mask >> curr_i) & 1): # curr_i must be in mask
                continue

            # prev_mask is the mask before segment curr_i was printed
            prev_mask = mask ^ (1 << curr_i)
            
            if prev_mask == 0: # curr_i was the first segment; already handled by base cases
                continue

            P_curr_i0 = segments_coords[curr_i][0]
            P_curr_i1 = segments_coords[curr_i][1]

            # To end at P_curr_i0 (end_type=0 for curr_i), print P_curr_i1 -> P_curr_i0.
            # Laser must travel to P_curr_i1 to start printing.
            start_point_for_curr_to_end_at_0 = P_curr_i1
            
            # To end at P_curr_i1 (end_type=1 for curr_i), print P_curr_i0 -> P_curr_i1.
            # Laser must travel to P_curr_i0 to start printing.
            start_point_for_curr_to_end_at_1 = P_curr_i0

            # Iterate over all possible previous segments (prev_i) in prev_mask
            for prev_i in range(N):
                if not ((prev_mask >> prev_i) & 1): # prev_i must be in prev_mask
                    continue
                
                # prev_i was the segment printed just before curr_i.
                P_prev_i0 = segments_coords[prev_i][0]
                P_prev_i1 = segments_coords[prev_i][1]

                # Case A: Laser was at P_prev_i0 (end_type=0 for prev_i) after printing prev_i
                if dp[prev_mask][prev_i][0] != float('inf'):
                    prev_laser_pos = P_prev_i0
                    
                    # Try to print curr_i ending at P_curr_i0
                    dist_travel = math.hypot(prev_laser_pos[0] - start_point_for_curr_to_end_at_0[0], 
                                             prev_laser_pos[1] - start_point_for_curr_to_end_at_0[1])
                    time_travel = dist_travel / float(S_speed)
                    dp[mask][curr_i][0] = min(dp[mask][curr_i][0], dp[prev_mask][prev_i][0] + time_travel)

                    # Try to print curr_i ending at P_curr_i1
                    dist_travel = math.hypot(prev_laser_pos[0] - start_point_for_curr_to_end_at_1[0], 
                                             prev_laser_pos[1] - start_point_for_curr_to_end_at_1[1])
                    time_travel = dist_travel / float(S_speed)
                    dp[mask][curr_i][1] = min(dp[mask][curr_i][1], dp[prev_mask][prev_i][0] + time_travel)

                # Case B: Laser was at P_prev_i1 (end_type=1 for prev_i) after printing prev_i
                if dp[prev_mask][prev_i][1] != float('inf'):
                    prev_laser_pos = P_prev_i1

                    # Try to print curr_i ending at P_curr_i0
                    dist_travel = math.hypot(prev_laser_pos[0] - start_point_for_curr_to_end_at_0[0], 
                                             prev_laser_pos[1] - start_point_for_curr_to_end_at_0[1])
                    time_travel = dist_travel / float(S_speed)
                    dp[mask][curr_i][0] = min(dp[mask][curr_i][0], dp[prev_mask][prev_i][1] + time_travel)

                    # Try to print curr_i ending at P_curr_i1
                    dist_travel = math.hypot(prev_laser_pos[0] - start_point_for_curr_to_end_at_1[0], 
                                             prev_laser_pos[1] - start_point_for_curr_to_end_at_1[1])
                    time_travel = dist_travel / float(S_speed)
                    dp[mask][curr_i][1] = min(dp[mask][curr_i][1], dp[prev_mask][prev_i][1] + time_travel)
                    
    # All N segments have been printed. The mask is (1<<N) - 1.
    final_mask = (1 << N) - 1
    min_total_travel_time = float('inf')
    
    # Find the minimum travel time by checking all possible last segments and their ending points
    for i in range(N): # i is the index of the last segment printed
        min_total_travel_time = min(min_total_travel_time, dp[final_mask][i][0])
        min_total_travel_time = min(min_total_travel_time, dp[final_mask][i][1])
        
    # The final answer is the minimum total travel time plus the fixed total printing time
    total_min_time = min_total_travel_time + total_printing_time_fixed_part
    
    # Print with required precision
    print(f"{total_min_time:.12f}")

solve()