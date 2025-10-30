import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))

    total_base_len = 0
    # diff_array[j] stores S_j - S_{j-1} (cyclically, S_0 = S_N)
    # We use 1-indexed bridges B_1, ..., B_N.
    # B_j connects island j and island (j%N)+1.
    # This means B_N connects N and 1.
    diff_array = [0] * (N + 2) # 1-indexed up to N. Index N+1 can be used for diff[v] when v=N+1.

    for k in range(M - 1):
        u, v_orig = X[k], X[k+1]
        
        # Normalize v to handle bridge indexing if v is destination of diff update
        # e.g. if path is u,...,N, current v is N. Next is 1.
        # diff[u]+=val, diff[v_target]-=val. v_target is node after end of path.
        # if path is 1->2->3, bridges B1,B2. u=1, v_orig=3. v_target=3.
        # diff[1]+=val, diff[3]-=val.
        
        # Calculate clockwise and counter-clockwise distances
        if u < v_orig:
            dist1 = v_orig - u  # Clockwise: u -> u+1 -> ... -> v_orig
        else: # u >= v_orig (u > v_orig or u == v_orig)
            dist1 = (N - u) + v_orig # Clockwise: u -> ... -> N -> 1 -> ... -> v_orig
        
        if u > v_orig:
            dist2 = u - v_orig # Counter-clockwise: u -> u-1 -> ... -> v_orig
        else: # u <= v_orig (u < v_orig or u == v_orig)
            dist2 = (N - v_orig) + u # Counter-clockwise: u -> ... -> 1 -> N -> ... -> v_orig

        shortest_dist = 0
        penalty = 0
        
        v_target_for_diff = 0 # The node that marks end of penalized bridge seq + 1

        if dist1 <= dist2:
            shortest_dist = dist1
            penalty = N - 2 * shortest_dist
            if penalty > 0:
                # Path is clockwise: u, u+1, ..., v_orig
                # Bridges are B_u, B_{u+1}, ..., B_{v_orig-1} (cyclically)
                # diff_array[u] increases, diff_array[v_orig] decreases
                diff_array[u] += penalty
                diff_array[v_orig] -= penalty
        else:
            shortest_dist = dist2
            penalty = N - 2 * shortest_dist
            if penalty > 0:
                # Path is counter-clockwise: u, u-1, ..., v_orig
                # Bridges are B_{u-1}, B_{u-2}, ..., B_{v_orig} (cyclically)
                # diff_array[v_orig] increases, diff_array[u] decreases
                diff_array[v_orig] += penalty
                diff_array[u] -= penalty
        
        total_base_len += shortest_dist

    S_N_true = 0
    for k in range(M - 1):
        u, v = X[k], X[k+1]
        
        if u < v:
            dist1 = v - u
            dist2 = (N - v) + u
        else: # u >= v
            dist1 = (N - u) + v
            dist2 = u - v
        
        penalty = 0
        bridge_N_on_shortest_path = False

        if dist1 <= dist2:
            penalty = N - 2 * dist1
            if u > v: # Clockwise path u...N...1...v includes bridge N
                bridge_N_on_shortest_path = True
        else:
            penalty = N - 2 * dist2
            if u < v: # Counter-clockwise path u...1...N...v includes bridge N
                bridge_N_on_shortest_path = True
        
        if penalty > 0 and bridge_N_on_shortest_path:
            S_N_true += penalty
            
    min_total_penalty_increase = S_N_true 
    current_S_val = S_N_true # This is S_N (penalty sum for bridge N)

    # Iterate from j=1 to N. `current_S_val` starts as S_N.
    # In iteration j, `current_S_val` becomes S_j.
    for j in range(1, N + 1):
        # `current_S_val` is S_{j-1} (or S_N if j=1)
        # `diff_array[j]` is S_j - S_{j-1}
        # So, new `current_S_val` will be S_j.
        current_S_val += diff_array[j]
        
        if j < N : # We are calculating S_1, ..., S_{N-1}. S_N is already accounted for in init.
             min_total_penalty_increase = min(min_total_penalty_increase, current_S_val)
        # If j == N, current_S_val is the re-calculated S_N. It should match S_N_true.
        # We don't need to min against it again as it was the initial value.
    
    print(total_base_len + min_total_penalty_increase)

solve()