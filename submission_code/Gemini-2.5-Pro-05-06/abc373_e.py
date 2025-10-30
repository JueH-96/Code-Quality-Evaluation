import sys

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    A_orig = list(map(int, sys.stdin.readline().split()))

    sum_A = sum(A_orig)
    R_total_remaining_votes = K - sum_A

    val_idx_list = sorted([(A_orig[i], i) for i in range(N)], key=lambda p: p[0], reverse=True)
    
    sorted_A_vals = [p[0] for p in val_idx_list]
    
    rank_of_orig_idx = [0] * N # rank_of_orig_idx[original_idx] = rank in sorted_A_vals
    for r, p_tuple in enumerate(val_idx_list):
        rank_of_orig_idx[p_tuple[1]] = r
        
    # pref_S[i] stores sum of sorted_A_vals[0]...sorted_A_vals[i-1]
    pref_S = [0] * (N + 1)
    for i in range(N):
        pref_S[i+1] = pref_S[i] + sorted_A_vals[i]

    def get_sum_from_sorted_prefix(start_idx, end_idx): # sum sorted_A_vals[start_idx...end_idx]
        if start_idx > end_idx or start_idx < 0 or end_idx >= N : # Basic safety, though logic should prevent invalid calls
             return 0
        return pref_S[end_idx + 1] - pref_S[start_idx]

    results = [-1] * N

    for k_orig_idx in range(N):
        Ak_val = A_orig[k_orig_idx] # Current votes of candidate k
        pos_k_in_sorted = rank_of_orig_idx[k_orig_idx] # Rank of candidate k in sorted_A_vals

        # Binary search for minimum X for candidate k_orig_idx
        
        # Check function: True if k_orig_idx wins with X_votes additional votes
        def check(X_votes_for_k):
            # Score of k if they get X_votes_for_k additional votes
            k_final_score = Ak_val + X_votes_for_k
            # Votes an adversary can distribute to other candidates
            votes_for_adversary = R_total_remaining_votes - X_votes_for_k
            
            # If k needs more votes than available, X_votes_for_k is impossible
            if votes_for_adversary < 0:
                return False 

            # C1 = count of candidates j (j != k_orig_idx) with A_orig[j] > k_final_score
            # Since X_votes_for_k >= 0, Ak_val <= k_final_score.
            # So k_orig_idx itself is not > k_final_score.
            # Thus, C1 is total count of candidates p with A_orig[p] > k_final_score.
            # Find p_U = index of first element in sorted_A_vals that is <= k_final_score
            # elements sorted_A_vals[0]...sorted_A_vals[p_U-1] are > k_final_score
            
            # Binary search for p_U
            p_U_bs_low, p_U_bs_high = 0, N - 1
            p_U = N # if all elements > k_final_score
            if N > 0: # handle N=0, though constraints say N>=1
                while p_U_bs_low <= p_U_bs_high:
                    mid = (p_U_bs_low + p_U_bs_high) // 2
                    if sorted_A_vals[mid] > k_final_score:
                        p_U_bs_low = mid + 1
                    else:
                        p_U = mid
                        p_U_bs_high = mid - 1
            
            C1 = p_U 

            if C1 >= M: # Too many others are already stronger
                return False

            # Adversary needs to make N_T more candidates stronger
            N_T_target_adversaries = M - C1 # N_T >= 1 because C1 < M

            # Candidates available for adversary (j != k_orig_idx, A_orig[j] <= k_final_score)
            # These are (N - p_U) candidates with scores <= k_final_score.
            # One of them is k_orig_idx. So (N - p_U - 1) others.
            num_available_for_adversary = (N - p_U) - 1
            
            if N == 1 : # Special case for N=1, num_available_for_adversary becomes -1
                 num_available_for_adversary = 0


            if num_available_for_adversary < N_T_target_adversaries:
                return True # Adversary doesn't have enough distinct candidates

            # Adversary picks N_T_target_adversaries from sorted_A_vals[p_U...]
            # skipping k_orig_idx if its rank pos_k_in_sorted falls in chosen range.
            
            sum_A_for_chosen_adversaries = 0
            # Range of candidates in sorted_A_vals for adversary to pick from initially:
            # [p_U, p_U + N_T_target_adversaries - 1]
            adv_pick_start_idx = p_U
            adv_pick_end_idx = p_U + N_T_target_adversaries - 1

            if pos_k_in_sorted >= adv_pick_start_idx and pos_k_in_sorted <= adv_pick_end_idx:
                # k_orig_idx is in the primary pool. Exclude it, take one more.
                sum_A_for_chosen_adversaries = get_sum_from_sorted_prefix(adv_pick_start_idx, adv_pick_end_idx)
                sum_A_for_chosen_adversaries -= Ak_val # Ak_val is sorted_A_vals[pos_k_in_sorted]
                # The next candidate to take is sorted_A_vals[adv_pick_end_idx + 1]
                # This index (p_U + N_T_target_adversaries) is valid because
                # num_available_for_adversary >= N_T_target_adversaries
                # (N - p_U) - 1 >= N_T_target_adversaries  => N - 1 >= p_U + N_T_target_adversaries
                # So index p_U + N_T_target_adversaries is at most N-1.
                sum_A_for_chosen_adversaries += sorted_A_vals[adv_pick_end_idx + 1]
            else:
                # k_orig_idx is not in the primary pool.
                sum_A_for_chosen_adversaries = get_sum_from_sorted_prefix(adv_pick_start_idx, adv_pick_end_idx)
            
            # Votes needed by adversary to make these N_T candidates have score k_final_score + 1
            votes_needed_by_adversary = (k_final_score + 1) * N_T_target_adversaries - sum_A_for_chosen_adversaries
            
            if votes_needed_by_adversary > votes_for_adversary:
                return True # Adversary cannot afford
            else:
                return False # Adversary can afford

        # Binary search for min X in [0, R_total_remaining_votes]
        bs_low, bs_high = 0, R_total_remaining_votes
        min_X_found = R_total_remaining_votes + 1

        while bs_low <= bs_high:
            mid_X = (bs_low + bs_high) // 2
            if mid_X < 0: # Should not happen with bs_low initialized to 0
                bs_low = mid_X + 1
                continue
            if check(mid_X):
                min_X_found = mid_X
                bs_high = mid_X - 1
            else:
                bs_low = mid_X + 1
        
        if min_X_found <= R_total_remaining_votes:
            results[k_orig_idx] = min_X_found
        else:
            # This means no X in [0, R_total_remaining_votes] worked
            results[k_orig_idx] = -1
            
    sys.stdout.write(" ".join(map(str, results)) + "
")

solve()