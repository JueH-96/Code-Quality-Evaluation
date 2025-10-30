import sys

sys.setrecursionlimit(4 * 10**5) # Increased recursion limit for deep DFS and CH computation queue
input = sys.stdin.readline

INF_COST_NUM = float('inf') # Using float('inf') for large costs in DP
INF_PICKS_SCALED = float('inf')
INF_PAIR = (INF_COST_NUM, INF_PICKS_SCALED)

# Memoization for the tree DP part of WQS
memo_dfs_dp = {}

# Adjacency list for children in rooted tree (rooted at 1)
children_adj = []

# Tree DP for WQS. u is current node. penalty_P/penalty_Q is lambda.
# Returns three pairs: (objective_value, picks_scaled_by_Q)
# objective_value = (sum_inactive_edges * Q) + (num_picks * P)
# picks_scaled_by_Q = num_picks * Q
# We lexicographically minimize (objective_value, picks_scaled_by_Q)
def dfs_dp(u, penalty_P, penalty_Q):
    state_tuple = (u, penalty_P, penalty_Q)
    if state_tuple in memo_dfs_dp:
        return memo_dfs_dp[state_tuple]

    # state 0: u's subtree unlit. u not picked. picks = 0.
    res0_obj_val = 0 
    res0_picks_scaled = 0
    for v, l_edge in children_adj[u]:
        v_dp0, _, _ = dfs_dp(v, penalty_P, penalty_Q)
        res0_obj_val += v_dp0[0] + l_edge * penalty_Q
        res0_picks_scaled += v_dp0[1]
    
    # state 1: u is lit, u is picked.
    res1_obj_val = penalty_P 
    res1_picks_scaled = penalty_Q 
    for v, l_edge in children_adj[u]:
        v_dp0, v_dp1, v_dp2 = dfs_dp(v, penalty_P, penalty_Q)
        
        cand_A = (v_dp0[0] + l_edge * penalty_Q, v_dp0[1]) # v unlit
        cand_B = (v_dp1[0], v_dp1[1]) # v lit, v picked
        cand_C = (v_dp2[0], v_dp2[1]) # v lit, v not picked

        best_child_val = cand_A
        if cand_B[0] < best_child_val[0] or \
           (cand_B[0] == best_child_val[0] and cand_B[1] < best_child_val[1]):
            best_child_val = cand_B
        if cand_C[0] < best_child_val[0] or \
           (cand_C[0] == best_child_val[0] and cand_C[1] < best_child_val[1]):
            best_child_val = cand_C
        
        res1_obj_val += best_child_val[0]
        res1_picks_scaled += best_child_val[1]

    # state 2: u is lit, u not picked. Some child must be lit.
    # Iteratively merge children.
    # D0: (obj_val, picks_scaled) if processed children are all unlit
    # D1: (obj_val, picks_scaled) if at least one processed child is lit
    current_D0_obj, current_D0_picks_s = 0, 0
    current_D1_obj, current_D1_picks_s = INF_PAIR 

    for v, l_edge in children_adj[u]:
        v_dp0, v_dp1, v_dp2 = dfs_dp(v, penalty_P, penalty_Q)
        
        next_D0_obj, next_D0_picks_s = INF_PAIR
        next_D1_obj, next_D1_picks_s = INF_PAIR

        # Calculate next_D0: current_D0 + child v unlit (v_dp0)
        if current_D0_obj != INF_COST_NUM:
            cost_v_unlit_obj = v_dp0[0] + l_edge * penalty_Q
            picks_v_unlit_s = v_dp0[1]
            next_D0_obj = current_D0_obj + cost_v_unlit_obj
            next_D0_picks_s = current_D0_picks_s + picks_v_unlit_s
        
        # Calculate next_D1
        # Case 1: current_D0 + child v is lit (v_dp1 or v_dp2)
        if current_D0_obj != INF_COST_NUM:
            # Child v lit by picking v
            cand1_obj = current_D0_obj + v_dp1[0]
            cand1_picks_s = current_D0_picks_s + v_dp1[1]
            if cand1_obj < next_D1_obj or \
               (cand1_obj == next_D1_obj and cand1_picks_s < next_D1_picks_s):
                next_D1_obj, next_D1_picks_s = cand1_obj, cand1_picks_s
            
            # Child v lit not by picking v
            cand2_obj = current_D0_obj + v_dp2[0]
            cand2_picks_s = current_D0_picks_s + v_dp2[1]
            if cand2_obj < next_D1_obj or \
               (cand2_obj == next_D1_obj and cand2_picks_s < next_D1_picks_s):
                next_D1_obj, next_D1_picks_s = cand2_obj, cand2_picks_s
        
        # Case 2: current_D1 + child v is any state
        if current_D1_obj != INF_COST_NUM:
            # Combine options for child v (same as in state 1 logic)
            cand_A = (v_dp0[0] + l_edge * penalty_Q, v_dp0[1])
            cand_B = (v_dp1[0], v_dp1[1])
            cand_C = (v_dp2[0], v_dp2[1])
            
            best_child_val_for_D1 = cand_A
            if cand_B[0] < best_child_val_for_D1[0] or \
               (cand_B[0] == best_child_val_for_D1[0] and cand_B[1] < best_child_val_for_D1[1]):
                best_child_val_for_D1 = cand_B
            if cand_C[0] < best_child_val_for_D1[0] or \
               (cand_C[0] == best_child_val_for_D1[0] and cand_C[1] < best_child_val_for_D1[1]):
                best_child_val_for_D1 = cand_C

            cand3_obj = current_D1_obj + best_child_val_for_D1[0]
            cand3_picks_s = current_D1_picks_s + best_child_val_for_D1[1]
            if cand3_obj < next_D1_obj or \
               (cand3_obj == next_D1_obj and cand3_picks_s < next_D1_picks_s):
                next_D1_obj, next_D1_picks_s = cand3_obj, cand3_picks_s
        
        current_D0_obj, current_D0_picks_s = next_D0_obj, next_D0_picks_s
        current_D1_obj, current_D1_picks_s = next_D1_obj, next_D1_picks_s
    
    res2_obj_val, res2_picks_scaled = current_D1_obj, current_D1_picks_s
    
    final_res0 = (res0_obj_val, res0_picks_scaled)
    final_res1 = (res1_obj_val, res1_picks_scaled)
    final_res2 = (res2_obj_val, res2_picks_scaled)
    
    memo_dfs_dp[state_tuple] = (final_res0, final_res1, final_res2)
    return final_res0, final_res1, final_res2

# Function to run DP and get (sum_inactive_edges, num_picks) for a given penalty P/Q
def get_dp_result_for_penalty(penalty_P, penalty_Q):
    global memo_dfs_dp
    memo_dfs_dp.clear() # Clear memo for new penalty
    
    # For root=1, it must be lit. So combine results of state1 and state2.
    _, r_dp1, r_dp2 = dfs_dp(1, penalty_P, penalty_Q)

    best_overall_obj_val = r_dp1[0]
    best_overall_picks_scaled = r_dp1[1]

    if r_dp2[0] < best_overall_obj_val or \
       (r_dp2[0] == best_overall_obj_val and r_dp2[1] < best_overall_picks_scaled):
        best_overall_obj_val = r_dp2[0]
        best_overall_picks_scaled = r_dp2[1]
    
    # Objective_val = sum_inactive_L_edges * Q + num_picks * P
    # Stored_picks_scaled = num_picks * Q
    # From these, derive sum_inactive_L_edges and num_picks
    if best_overall_picks_scaled == INF_PICKS_SCALED : # No valid way to pick
        return INF_COST_NUM, -1 # Or some indicator of failure
        
    num_picks = best_overall_picks_scaled // penalty_Q
    sum_inactive = (best_overall_obj_val - num_picks * penalty_P) // penalty_Q
    
    return sum_inactive, num_picks

def solve():
    N = int(input())
    adj = [[] for _ in range(N + 1)]
    total_edge_len = 0
    for _ in range(N - 1):
        u, v, l = map(int, input().split())
        adj[u].append((v, l))
        adj[v].append((u, l))
        total_edge_len += l

    if N == 1:
        print(0)
        return

    global children_adj 
    children_adj = [[] for _ in range(N + 1)]
    
    # BFS to build rooted tree structure (children_adj)
    q_bfs = [(1, 0)] # node, parent
    head_bfs = 0
    visited_bfs = [False] * (N + 1)
    visited_bfs[1] = True
    
    # For N_L_eff calculation
    num_actual_children = [0] * (N+1)

    while head_bfs < len(q_bfs):
        u, p = q_bfs[head_bfs]
        head_bfs += 1
        
        for v, l_edge in adj[u]:
            if v == p:
                continue
            if visited_bfs[v]: continue # Should not happen
            visited_bfs[v] = True
            
            children_adj[u].append((v, l_edge))
            num_actual_children[u]+=1
            q_bfs.append((v, u))

    n_l_eff = 0
    if N > 1:
        for i in range(1, N + 1):
            if num_actual_children[i] == 0: # It's a leaf in the rooted tree
                if i != 1: # Leaf is not root itself
                    n_l_eff += 1
        if n_l_eff == 0 : # Only root 1 has no children (N=1) or all leaves are children of root 1, and root 1 is not a leaf.
                         # Or, root 1 is a leaf and has one child (N=2, 1-2).
                         # if root 1 is the only node or all other nodes are its direct children and become leaves.
                         # E.g. star graph 1-2, 1-3, 1-4. Leaves 2,3,4. n_l_eff=3.
                         # E.g. path 1-2-3. Leaves 3. n_l_eff=1.
                         # E.g. path 1-2. Leaf 2. n_l_eff=1.
            n_l_eff = 1 # If no leaves other than root, or tree is small, effectively need 1 pick unless N=1.

    min_inactive_sum = [-1] * (max(N,2) + 1) # Stores f(K)

    min_inactive_sum[0] = total_edge_len 
    if n_l_eff > 0:
         min_inactive_sum[n_l_eff] = 0

    recursion_q = []
    if n_l_eff > 0: 
        recursion_q.append((0, n_l_eff))

    processed_k_in_hull = {0}
    if n_l_eff > 0: processed_k_in_hull.add(n_l_eff)


    while recursion_q:
        k_low, k_high = recursion_q.pop(0)

        if k_low >= k_high -1 : 
            continue
        
        # Penalty P/Q = (f(k_low) - f(k_high)) / (k_high - k_low)
        # This is -slope. We use this as penalty lambda.
        penalty_P = min_inactive_sum[k_low] - min_inactive_sum[k_high]
        penalty_Q = k_high - k_low

        # If P becomes negative, it means convexity is violated or k_high was a bad point.
        # This implies f(k_low) < f(k_high). This should not happen if f is convex non-increasing.
        # A P=0 implies a flat segment.
        if penalty_P < 0: penalty_P = 0 # Force non-negative penalty; effectively lambda >= 0

        current_sum_inactive, k_opt = get_dp_result_for_penalty(penalty_P, penalty_Q)
        
        if k_opt == -1 or current_sum_inactive == INF_COST_NUM : # DP failed or no valid picks
            # This segment might be problematic or fully explored by endpoints
            continue

        # Check if this k_opt offers a better (lower) sum_inactive value or same value with fewer picks
        # Or if it's a new point for K.
        if min_inactive_sum[k_opt] == -1 or \
           current_sum_inactive < min_inactive_sum[k_opt] or \
           (current_sum_inactive == min_inactive_sum[k_opt] and k_opt < k_opt): # Redundant k_opt<k_opt, but shows idea
            min_inactive_sum[k_opt] = current_sum_inactive
        
        # Add k_opt to list of known CH vertices whether it updated or not,
        # to ensure segments are explored.
        processed_k_in_hull.add(k_opt)


        # Recurse on sub-ranges
        # Only recurse if k_opt is strictly between k_low and k_high for this slope.
        # If k_opt is an endpoint (k_low or k_high), it means no new CH vertex was found
        # strictly between them for this slope. The segment (k_low, k_high) is a line on CH.
        if k_opt > k_low and k_opt < k_high :
             recursion_q.append((k_low, k_opt))
             recursion_q.append((k_opt, k_high))
        elif k_opt == k_low : # Found k_low as optimal for this slope. Explore (k_low, k_high) -> (k_opt, k_high)
             recursion_q.append((k_low, k_high)) # This might lead to infinite loop if not careful
                                                 # Instead, if k_opt is an endpoint, no new point strictly between was found
                                                 # So (k_low, k_high) is an edge of CH. It will be interpolated.
                                                 # The issue is that k_opt can be outside [k_low, k_high]
                                                 # or be one of k_low, k_high. The CH search needs care.
                                                 # A common way: if k_opt is already known (min_inactive_sum[k_opt]!=-1),
                                                 # and val is not better, skip. Else update and recurse.
             pass # This segment is likely a line on CH. Interpolation will handle it.
        elif k_opt == k_high :
             pass # Same as above.

    # Interpolate for K values not on CH vertices explicitly found
    ch_vertices_k = sorted(list(processed_k_in_hull))
    
    final_ch_vertices = []
    if ch_vertices_k:
        final_ch_vertices.append(ch_vertices_k[0])
        for i in range(1,len(ch_vertices_k)):
            if ch_vertices_k[i] > ch_vertices_k[i-1]: # Ensure strictly increasing
                 if min_inactive_sum[ch_vertices_k[i]] != -1 : # Only use valid computed points
                    final_ch_vertices.append(ch_vertices_k[i])
    ch_vertices_k = final_ch_vertices
    
    # If only [0] or [0, n_l_eff] are in ch_vertices_k, means straight line from f(0) to f(n_l_eff)
    # This case implies min_inactive_sum for intermediate K values might be -1.

    for i in range(len(ch_vertices_k) - 1):
        k_prev, k_next = ch_vertices_k[i], ch_vertices_k[i+1]
        
        # Ensure values for k_prev, k_next are actually computed
        if min_inactive_sum[k_prev] == -1 or min_inactive_sum[k_next] == -1: continue

        f_prev, f_next = min_inactive_sum[k_prev], min_inactive_sum[k_next]
        
        if k_prev >= k_next -1 : continue 

        for k_interp in range(k_prev + 1, k_next):
            if min_inactive_sum[k_interp] != -1: continue # Already computed, likely a CH vertex itself.
            
            numerator = f_prev * (k_next - k_interp) + f_next * (k_interp - k_prev)
            denominator = k_next - k_prev
            if denominator == 0: continue # Should not happen if k_prev < k_next
            min_inactive_sum[k_interp] = numerator // denominator
    
    ans_scores = [0] * (N + 1)
    for k_iter in range(1, N + 1):
        current_f_k = -1
        if k_iter >= n_l_eff and n_l_eff > 0:
             current_f_k = 0
        elif N == 1 and k_iter == 1: # Score 0 for N=1, K=1
             current_f_k = total_edge_len # f(K)=L_total implies score 0. This is correct.
        else:
             if k_iter < len(min_inactive_sum) and min_inactive_sum[k_iter] != -1:
                current_f_k = min_inactive_sum[k_iter]
             else: # Should not happen if interpolation is correct and covers all K up to n_l_eff
                  # Fallback, or error. This might mean k_iter > n_l_eff and n_l_eff=0.
                  # This implies k_iter > 0 for N=1, only k_iter=1 possible. Score 0. f(1)=L_total=0
                  if N==1: current_f_k = 0
                  else: # This indicates an issue, K not covered. Maybe default to 2*L_total?
                        # Or if k_iter is large enough, f(K)=0
                        # This means k_iter > n_l_eff (which might be 0 or 1)
                        current_f_k = 0


        ans_scores[k_iter] = 2 * (total_edge_len - current_f_k)
    
    for k_res in range(1, N + 1):
        print(ans_scores[k_res])

solve()