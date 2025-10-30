import sys

def solve():
    N = int(sys.stdin.readline())
    weights_list = list(map(int, sys.stdin.readline().split()))
    
    intervals_orig = []
    for i in range(N):
        l, r = map(int, sys.stdin.readline().split())
        intervals_orig.append({'L': l, 'R': r, 'W': weights_list[i], 'id': i})

    # Max possible coordinate value for R_i can be 2N.
    # Array indices for coordinates need to handle up to this value.
    # Let MAX_COORD_VAL be the max coordinate encountered or simply 2*N.
    # For safety and 1-based indexing convenience, use a slightly larger size.
    # Max R_i is 2N. Max L_i is 2N.
    # Range of coordinates is [1, 2N]. Let's use array size up to 2N+1 for 1-based coords.
    MAX_COORD_ARRAY_SIZE = 2 * N + 2 

    # Check for global intersection point
    max_L_global = 0
    min_R_global = float('inf')
    for i in range(N):
        max_L_global = max(max_L_global, intervals_orig[i]['L'])
        min_R_global = min(min_R_global, intervals_orig[i]['R'])
    
    if max_L_global <= min_R_global:
        Q = int(sys.stdin.readline())
        results = []
        for _ in range(Q):
            s, t = map(int, sys.stdin.readline().split()) # Read s,t even if not used
            results.append("-1")
        sys.stdout.write("
".join(results) + "
")
        return

    # Precompute prefix minimums for R_x <= k
    # pmr_min_W_le_R[k] = min W_i over intervals i where R_i <= k
    pmr_min_W_le_R = [float('inf')] * MAX_COORD_ARRAY_SIZE
    min_W_at_R_coord = [float('inf')] * MAX_COORD_ARRAY_SIZE
    for i_idx in range(N):
        interval = intervals_orig[i_idx]
        min_W_at_R_coord[interval['R']] = min(min_W_at_R_coord[interval['R']], interval['W'])
    
    current_min_for_pmr = float('inf')
    for k in range(1, MAX_COORD_ARRAY_SIZE): # Iterate from coordinate 1
        current_min_for_pmr = min(current_min_for_pmr, min_W_at_R_coord[k])
        pmr_min_W_le_R[k] = current_min_for_pmr

    # Precompute suffix minimums for L_x >= k
    # sml_min_W_ge_L[k] = min W_i over intervals i where L_i >= k
    sml_min_W_ge_L = [float('inf')] * MAX_COORD_ARRAY_SIZE
    min_W_at_L_coord = [float('inf')] * MAX_COORD_ARRAY_SIZE
    for i_idx in range(N):
        interval = intervals_orig[i_idx]
        min_W_at_L_coord[interval['L']] = min(min_W_at_L_coord[interval['L']], interval['W'])
        
    current_min_for_sml = float('inf')
    # Iterate from max possible coord down to 1
    for k in range(MAX_COORD_ARRAY_SIZE - 1, 0, -1): 
        current_min_for_sml = min(current_min_for_sml, min_W_at_L_coord[k])
        sml_min_W_ge_L[k] = current_min_for_sml
    
    results = []
    Q_count = int(sys.stdin.readline())

    for _ in range(Q_count):
        s_orig_idx, t_orig_idx = map(int, sys.stdin.readline().split())
        s_idx = s_orig_idx - 1 # 0-indexed
        t_idx = t_orig_idx - 1 # 0-indexed

        s_interval = intervals_orig[s_idx]
        t_interval = intervals_orig[t_idx]

        current_ans = float('inf')

        # Path length 1: direct edge s-t
        if s_interval['R'] < t_interval['L'] or t_interval['R'] < s_interval['L']:
            current_ans = min(current_ans, s_interval['W'] + t_interval['W'])
        
        # Path length 2: s-x-t
        # Cost W_s + W_t + W_x
        # Case A: R_x < min(L_s, L_t)
        limit_coord_case_A = min(s_interval['L'], t_interval['L']) - 1
        if limit_coord_case_A >= 1: # Check if coordinate is valid (>=1)
            min_w_x_case_A = pmr_min_W_le_R[limit_coord_case_A]
            if min_w_x_case_A != float('inf'):
                current_ans = min(current_ans, s_interval['W'] + t_interval['W'] + min_w_x_case_A)
        
        # Case B: L_x > max(R_s, R_t)
        limit_coord_case_B = max(s_interval['R'], t_interval['R']) + 1
        if limit_coord_case_B < MAX_COORD_ARRAY_SIZE: # Check if coordinate is valid (< MAX_COORD_ARRAY_SIZE)
            min_w_x_case_B = sml_min_W_ge_L[limit_coord_case_B]
            if min_w_x_case_B != float('inf'):
                current_ans = min(current_ans, s_interval['W'] + t_interval['W'] + min_w_x_case_B)
        
        if current_ans == float('inf'):
            results.append("-1")
        else:
            results.append(str(current_ans))
            
    sys.stdout.write("
".join(results) + "
")

solve()