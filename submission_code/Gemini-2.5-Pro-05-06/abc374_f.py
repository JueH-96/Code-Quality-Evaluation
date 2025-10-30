import math

def solve():
    N, K, X = map(int, input().split())
    T_values = list(map(int, input().split())) # These are 0-indexed T_values[0]...T_values[N-1]

    # Prefix sums for T_values
    # P_T[k] stores sum(T_values[0]...T_values[k-1]), so P_T[0]=0
    P_T = [0] * (N + 1)
    for i in range(N):
        P_T[i+1] = P_T[i] + T_values[i]

    # dp[i] stores list of (total_dissatisfaction, ship_day_of_last_batch)
    # for shipping first i orders (T_values[0]...T_values[i-1])
    dp = [[] for _ in range(N + 1)]

    # Base case: dp[0] for 0 orders shipped.
    # initial_prev_ship_day ensures that (initial_prev_ship_day + X) does not
    # make the first shipment day later than T_values[0] if it's not necessary.
    # T_values[0] - X means (initial_prev_ship_day + X) = T_values[0].
    # Since all T_values >= 1, T_values[0]-X can be negative, which is fine.
    initial_prev_ship_day = T_values[0] - X 
    dp[0] = [(0, initial_prev_ship_day)]

    # Iterate for dp[1]...dp[N]
    # dp[i] considers orders T_values[0]...T_values[i-1]
    for i in range(1, N + 1):
        candidate_pairs_for_dp_i = []
        
        # Last batch includes orders from index actual_j_start to i-1 (0-indexed)
        # Let 1-indexed problem orders be $j_{pb} \dots i_{pb}$.
        # This corresponds to 0-indexed $T[(j_{pb}-1) \dots (i_{pb}-1)]$.
        # Here, $i_{pb} = i$. So batch is $T[(j_{pb}-1) \dots (i-1)]$.
        # The number of items is $i - (j_{pb}-1)$. Max K.
        # So $i - (j_{pb}-1) \le K \implies j_{pb}-1 \ge i-K \implies j_{pb} \ge i-K+1$.
        # $j_{pb}$ also $\le i$.
        # actual_j_start = $j_{pb}-1$.
        for j_pb_start_loop in range(max(1, i - K + 1), i + 1):
            actual_j_start_idx = j_pb_start_loop - 1 # 0-indexed start of current batch
            
            num_items_in_batch = i - actual_j_start_idx
            
            # Latest placement time for this batch (orders actual_j_start_idx to i-1)
            # is T_values[i-1] because T_values is sorted.
            latest_placement_time_for_batch = T_values[i-1]
            
            # Sum of T_p for p in this batch.
            # Orders from T_values[actual_j_start_idx] to T_values[i-1].
            # Sum is P_T[i] - P_T[actual_j_start_idx].
            sum_T_for_batch = P_T[i] - P_T[actual_j_start_idx]

            # These orders are processed after orders T_values[0]...T_values[actual_j_start_idx-1]
            # So we look at dp[actual_j_start_idx] for previous states.
            for prev_total_dissat, prev_ship_day in dp[actual_j_start_idx]:
                current_ship_day = max(latest_placement_time_for_batch, prev_ship_day + X)
                
                batch_dissat = num_items_in_batch * current_ship_day - sum_T_for_batch
                
                new_total_dissat = prev_total_dissat + batch_dissat
                candidate_pairs_for_dp_i.append((new_total_dissat, current_ship_day))
        
        candidate_pairs_for_dp_i.sort() 

        pruned_dp_i_list = []
        min_S_among_kept_pairs = float('inf') 
        
        if not candidate_pairs_for_dp_i and i > 0 : # Should not be empty if solution exists
             pass # dp[i] will be empty

        for d_cand, s_cand in candidate_pairs_for_dp_i:
            if pruned_dp_i_list and pruned_dp_i_list[-1][0] == d_cand:
                # This (d_cand, s_cand) has same D as last kept pair.
                # Sorting ensures s_cand >= S of last kept pair (which was minimal S for that D).
                # So this one is dominated or identical. Skip.
                continue 
            
            if s_cand < min_S_among_kept_pairs:
                # This (d_cand, s_cand) has a new D value (d_cand > D of last kept)
                # OR it's the first pair being added.
                # If s_cand is also better (smaller) than S of last kept pair, it's non-dominated.
                pruned_dp_i_list.append((d_cand, s_cand))
                min_S_among_kept_pairs = s_cand
        
        dp[i] = pruned_dp_i_list

    min_overall_total_dissatisfaction = float('inf')
    # Constraints: N >= 1, so dp[N] should not be empty.
    if not dp[N] : # Should not happen given N >= 1
        min_overall_total_dissatisfaction = 0
    else:
        for d_final, _ in dp[N]: # s_final is not needed for the final answer
            min_overall_total_dissatisfaction = min(min_overall_total_dissatisfaction, d_final)
            
    print(min_overall_total_dissatisfaction)

solve()