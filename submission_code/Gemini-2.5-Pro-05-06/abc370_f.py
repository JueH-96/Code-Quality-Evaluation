import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    if K == N:
        min_mass = A[0]
        for i in range(1, N):
            if A[i] < min_mass:
                min_mass = A[i]
        print(min_mass, 0)
        return

    LOGK = (K - 1).bit_length()
    if K == 0 : LOGK = 0 # Should not happen based on constraints K>=2
    if K == 1 : LOGK = 0 # K-1 = 0, bit_length is 0. max_k_bit for path decomp will be 0.

    # Binary lifting tables
    dp_f = [[0] * N for _ in range(LOGK + 1)] # Stores next node
    dp_g = [[0] * N for _ in range(LOGK + 1)] # Stores sum of pieces counts
    dp_h = [[False] * N for _ in range(LOGK + 1)] # Stores if any segment on path has sum == x

    # Temporary arrays for precomputation within check(M) or for part 2
    cnt_arr_temp = [0] * N # Number of pieces for a segment
    val_arr_temp = [0] * N # Sum of masses for a segment
    next_node_arr_temp = [0] * N # Start of next segment

    memo_can_M_res = {} # Memoization for check(M) results - not strictly needed with Python's default recursion limit for binary search

    total_sum_A = sum(A)

    def precompute_arrays_for_M(M_target):
        if M_target > total_sum_A : return False # Optimization: if M is larger than total sum, impossible

        current_sum = 0
        r = 0
        # Initial window for start_idx = 0
        for i in range(N): # Max N pieces for first segment
            current_sum += A[r % N]
            r += 1
            if current_sum >= M_target:
                break
        
        if current_sum < M_target: # M_target is too large, even all N pieces sum to < M_target
            return False

        # Sliding window for all start_idx
        for l in range(N):
            if l > 0:
                current_sum -= A[(l - 1 + N) % N] # Correctly handle circular index for l-1
            
            # Expand window from right if necessary. Max N pieces in a segment.
            # (r-l) gives current segment length.
            while current_sum < M_target:
                if (r - l) >= N: # Already taken all N pieces for this segment
                    # This means sum of all N pieces < M_target, should have been caught by current_sum < M_target check after initial window
                    # Or by M_target > total_sum_A check.
                    return False 
                current_sum += A[r % N]
                r += 1
            
            cnt_arr_temp[l] = r - l
            val_arr_temp[l] = current_sum
            next_node_arr_temp[l] = r % N
        return True

    def check(M_check):
        if M_check == 0: return True 
        
        possible_to_form = precompute_arrays_for_M(M_check)
        if not possible_to_form:
            return False

        for i in range(N):
            dp_f[0][i] = next_node_arr_temp[i]
            dp_g[0][i] = cnt_arr_temp[i]
            # dp_h is not used in check(M_check)

        for k_lift in range(LOGK):
            for i in range(N):
                next_f = dp_f[k_lift][i]
                dp_f[k_lift+1][i] = dp_f[k_lift][next_f]
                
                val_g1 = dp_g[k_lift][i]
                val_g2 = dp_g[k_lift][next_f]

                # Cap total pieces to N+1 if it exceeds N. Any path using > N pieces is invalid.
                if val_g1 > N or val_g2 > N or val_g1 + val_g2 > N:
                    dp_g[k_lift+1][i] = N + 1
                else:
                    dp_g[k_lift+1][i] = val_g1 + val_g2
        
        min_total_pieces_K_segments = N + 1 
        for i in range(N): 
            current_node = i
            current_total_pieces = 0
            for k_bit in range(LOGK, -1, -1):
                if (K >> k_bit) & 1:
                    current_total_pieces += dp_g[k_bit][current_node]
                    current_node = dp_f[k_bit][current_node]
                    if current_total_pieces > N: 
                        break 
            if current_total_pieces <= N:
                 min_total_pieces_K_segments = current_total_pieces # Found a valid way
                 break # Optimization: if one s_0 works, M_check is possible.
        
        return (min_total_pieces_K_segments <= N)

    # Binary search for x
    low_bs, high_bs = 1, total_sum_A 
    x_max_min_sum = 0
    while low_bs <= high_bs:
        mid_bs = (low_bs + high_bs) // 2
        if mid_bs == 0 : # Should not happen with low_bs starting at 1
            low_bs = 1
            continue
        if check(mid_bs):
            x_max_min_sum = mid_bs
            low_bs = mid_bs + 1
        else:
            high_bs = mid_bs - 1
    
    # Part 2: Calculate y
    precompute_arrays_for_M(x_max_min_sum) # Use x_max_min_sum as M_target

    for i in range(N):
        dp_f[0][i] = next_node_arr_temp[i]
        dp_g[0][i] = cnt_arr_temp[i]
        dp_h[0][i] = (val_arr_temp[i] == x_max_min_sum)

    for k_lift in range(LOGK):
        for i in range(N):
            next_f = dp_f[k_lift][i]
            dp_f[k_lift+1][i] = dp_f[k_lift][next_f]
            
            val_g1 = dp_g[k_lift][i]
            val_g2 = dp_g[k_lift][next_f]
            if val_g1 > N or val_g2 > N or val_g1 + val_g2 > N:
                 dp_g[k_lift+1][i] = N + 1
            else:
                dp_g[k_lift+1][i] = val_g1 + val_g2
            
            dp_h[k_lift+1][i] = dp_h[k_lift][i] or dp_h[k_lift][next_f]

    is_ever_cut = [False] * N
    
    for i in range(N): 
        current_node_for_len_calc = i
        len_K_segments = 0
        for k_bit in range(LOGK, -1, -1):
            if (K >> k_bit) & 1:
                len_K_segments += dp_g[k_bit][current_node_for_len_calc]
                current_node_for_len_calc = dp_f[k_bit][current_node_for_len_calc]
                if len_K_segments > N: break
        
        if len_K_segments > N:
            continue

        # Check if this s_0=i is in Q
        valid_for_Q = False
        num_segments_to_check_for_x = K
        if len_K_segments < N: # Slack exists
            if K > 1 : # If K=1 and slack, it means total sum != x which is impossible as x is total_sum for K=1
                       # So if K=1, len_K_segments will be N if sum(A)=x.
                num_segments_to_check_for_x = K - 1
            # If K=1 and slack > 0: this implies N - L(s0,x) > 0. L(s0,x) is cnt[s0]. So cnt[s0] < N.
            # This means the single segment did not use all pieces. This is not allowed for K=1.
            # The setup of L(s0,M) <= N implies that for K=1, cnt[s0] must be N.
            # So for K=1, len_K_segments will always be N. The slack case N-L>0 is not for K=1.
            # This part is fine. If K=1, num_segments_to_check_for_x remains K=1.

        if num_segments_to_check_for_x == 0: # K=1 and slack case, means we only check the first segment.
                                             # If K=1, N-L(s0,x)>0, then num_segments_to_check_for_x becomes 0.
                                             # But if K=1, first segment is the only segment. If it has slack, means val[s0] must be x.
                                             # The path length K for any_K_seg_sum_is_x should be used.
            if K == 1 and len_K_segments < N: # This state means cnt_arr_temp[i] < N.
                                              # This case should not be a valid division for K=1.
                                              # $L(s_0, M) \le N$ implies $cnt[s_0] \le N$. For $K=1$, it must be $cnt[s_0]=N$.
                                              # So for K=1, len_K_segments must be N. The slack logic is for K>1.
                                              # The problem statement implies $K \ge 2$. My K=N case is correct. This is fine.
                 pass # This configuration for K=1 with slack is invalid. valid_for_Q remains False.

        path_node = i
        any_seg_sum_is_x = False
        # Max_k_bit for path length num_segments_to_check_for_x
        # Use LOGK as upper bound for k_bit, it's fine.
        # temp_K_path = num_segments_to_check_for_x
        for k_bit in range(LOGK, -1, -1):
            if (num_segments_to_check_for_x >> k_bit) & 1:
                any_seg_sum_is_x = any_seg_sum_is_x or dp_h[k_bit][path_node]
                path_node = dp_f[k_bit][path_node]
        
        if any_seg_sum_is_x:
            valid_for_Q = True
        
        if valid_for_Q:
            temp_curr_node = i
            for _ in range(K): # Mark K cuts
                cut_idx = (temp_curr_node - 1 + N) % N
                is_ever_cut[cut_idx] = True
                if _ < K -1 : # Avoid one extra jump for last segment for loop counting.
                   temp_curr_node = dp_f[0][temp_curr_node] 
    
    y = N - sum(is_ever_cut)
    print(x_max_min_sum, y)

solve()