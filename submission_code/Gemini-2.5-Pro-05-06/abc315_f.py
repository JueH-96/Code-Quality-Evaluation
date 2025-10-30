import math
import sys

def solve():
    N = int(sys.stdin.readline())
    points_coords = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        points_coords.append((x, y))

    memo_dist = {}
    def get_dist(p1_original_idx, p2_original_idx): # Takes 1-indexed original checkpoint numbers
        p1_list_idx = p1_original_idx - 1
        p2_list_idx = p2_original_idx - 1
        
        key_tuple = tuple(sorted((p1_list_idx, p2_list_idx))) # Canonical key for memoization
        
        if key_tuple in memo_dist:
            return memo_dist[key_tuple]

        x1, y1 = points_coords[p1_list_idx]
        x2, y2 = points_coords[p2_list_idx]
        
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        memo_dist[key_tuple] = dist
        return dist

    C_L_CONST = 35 # Max number of skips to track, based on penalty growth analysis
    
    # dp[i][c]: min travel distance to reach checkpoint P_i (1-indexed),
    # having skipped 'c' checkpoints totally from {P_2, ..., P_{i-1}}.
    dp = [[float('inf')] * (C_L_CONST + 1) for _ in range(N + 1)]

    dp[1][0] = 0.0 # Base case: at P_1, 0 skips, 0 distance
    
    for i in range(2, N + 1): # Current checkpoint P_i (1-indexed)
        # c_total_skipped_before_i: total skips among {P_2, ..., P_{i-1}}
        # Max possible skips before P_i is i-2. Also limited by C_L_CONST.
        max_c_for_this_i = min(i - 2, C_L_CONST)

        for c_total_skipped_before_i in range(max_c_for_this_i + 1):
            # k_skipped_segment: skips between P_prev_idx and P_i
            # P_prev_idx = i - 1 - k_skipped_segment
            max_k_for_this_c_and_i = min(c_total_skipped_before_i, i - 2)
            # (i-2 is max k if P_prev_idx = P_1)

            for k_skipped_segment in range(max_k_for_this_c_and_i + 1):
                prev_checkpoint_original_idx = i - 1 - k_skipped_segment
                c_skipped_before_prev = c_total_skipped_before_i - k_skipped_segment
                
                if dp[prev_checkpoint_original_idx][c_skipped_before_prev] == float('inf'):
                    continue 
                
                dist_val = get_dist(prev_checkpoint_original_idx, i)
                
                current_total_dist = dp[prev_checkpoint_original_idx][c_skipped_before_prev] + dist_val
                dp[i][c_total_skipped_before_i] = min(dp[i][c_total_skipped_before_i], current_total_dist)
    
    min_overall_s = float('inf')

    # Calculate final answer: min(dp[N][c] + penalty(c))
    # Max skips for P_N is N-2. Also limited by C_L_CONST.
    max_total_skips_at_N = min(N - 2, C_L_CONST)
    if N < 2: # Should not happen given N >= 2 constraint
        max_total_skips_at_N = 0 
    elif N == 2: # Checkpoints P_1, P_2. N-2 = 0 skips.
        max_total_skips_at_N = 0


    for c in range(max_total_skips_at_N + 1):
        cost_to_reach_N_with_c_skips = dp[N][c]

        if cost_to_reach_N_with_c_skips == float('inf'):
            continue

        penalty_val = 0.0
        if c > 0:
            penalty_val = float(1 << (c - 1)) # 2^(c-1)
        
        current_s = cost_to_reach_N_with_c_skips + penalty_val
        min_overall_s = min(min_overall_s, current_s)
        
    sys.stdout.write(f"{min_overall_s:.12f}
")

solve()