import sys

def main():
    MOD = 998244353

    # Precompute P_idx to P_vector mapping
    P_idx_to_vec_map = []
    for i in range(8):
        P_idx_to_vec_map.append(((i >> 2) & 1, (i >> 1) & 1, i & 1))

    # Memoization dictionaries for helper functions (minor optimization for clarity/safety)
    memo_P_to_idx = {}
    memo_P_to_Q = {}
    memo_Q_to_idx = {}

    def P_to_idx(p_vec):
        # p_vec is (pa, pb, pc)
        if p_vec in memo_P_to_idx:
            return memo_P_to_idx[p_vec]
        res = p_vec[0] * 4 + p_vec[1] * 2 + p_vec[2]
        memo_P_to_idx[p_vec] = res
        return res

    def idx_to_P(p_idx):
        return P_idx_to_vec_map[p_idx]

    def P_to_Q(p_vec):
        # p_vec is (pa, pb, pc)
        # Returns (q1, q2) where q1 = pa^pb, q2 = pb^pc
        if p_vec in memo_P_to_Q:
            return memo_P_to_Q[p_vec]
        q1 = p_vec[0] ^ p_vec[1]
        q2 = p_vec[1] ^ p_vec[2]
        res = (q1, q2)
        memo_P_to_Q[p_vec] = res
        return res
    
    def Q_to_idx(q_vec):
        # q_vec is (q1, q2)
        if q_vec in memo_Q_to_idx:
            return memo_Q_to_idx[q_vec]
        res = q_vec[0] * 2 + q_vec[1]
        memo_Q_to_idx[q_vec] = res
        return res

    # Initial values for k=0 (empty prefix)
    P0_vec = (0, 0, 0)  # Parities for empty string
    P0_idx = P_to_idx(P0_vec)
    Q0_vec = P_to_Q(P0_vec) # (0,0)
    Q0_idx = Q_to_idx(Q0_vec)
    
    # Config tuple for Q(0): (count for Q_idx=0, Q_idx=1, Q_idx=2, Q_idx=3)
    # Q(0) is Q0_vec, so its count is 1. Others are 0.
    initial_config_list = [0, 0, 0, 0]
    initial_config_list[Q0_idx] = 1
    initial_config_tuple = tuple(initial_config_list)
    
    N, K_limit = map(int, sys.stdin.readline().split())
    S_chars = sys.stdin.readline().strip()
    
    # dp_curr: map from (good_count, P_idx, config_tuple) to ways
    dp_curr = {}
    # State for k=0 (empty prefix): 0 good_substrings, P(0) is P0_idx, config for Q(0)
    dp_curr[(0, P0_idx, initial_config_tuple)] = 1

    # k_loop_var is current length of prefix processed. Iterate N times to process S[0]...S[N-1].
    for k_loop_var in range(N): 
        dp_next = {} # Stores states for prefix of length k_loop_var + 1
        
        current_char_options = []
        char_at_k = S_chars[k_loop_var]
        if char_at_k == '?':
            current_char_options = ['A', 'B', 'C']
        else:
            current_char_options = [char_at_k]

        for prev_state_tuple, ways in dp_curr.items():
            prev_good_count, prev_P_idx, prev_config_tuple = prev_state_tuple
            
            prev_P_vec = idx_to_P(prev_P_idx) # P(k_loop_var)
            
            for char_val_choice in current_char_options:
                # Calculate P(k_loop_var + 1)
                next_P_vec_list_temp = list(prev_P_vec) # Modifiable copy
                if char_val_choice == 'A':
                    next_P_vec_list_temp[0] ^= 1
                elif char_val_choice == 'B':
                    next_P_vec_list_temp[1] ^= 1
                else: # char_val_choice == 'C'
                    next_P_vec_list_temp[2] ^= 1
                next_P_vec = tuple(next_P_vec_list_temp) # P(k_loop_var + 1)
                next_P_idx = P_to_idx(next_P_vec)

                # Calculate Q(P(k_loop_var + 1))
                next_Q_vec = P_to_Q(next_P_vec)
                next_Q_idx = Q_to_idx(next_Q_vec)
                
                # Number of good substrings ending at current position (k_loop_var + 1)
                # These are S[j .. k_loop_var+1] where Q(P(k_loop_var+1)) == Q(P(j-1))
                # prev_config_tuple stores counts for Q(P(0))...Q(P(k_loop_var))
                newly_good_substrings = prev_config_tuple[next_Q_idx]
                
                current_total_good_count = prev_good_count + newly_good_substrings
                # Cap at K_limit for DP state
                actual_good_count_for_dp = min(K_limit, current_total_good_count)

                # Update config tuple for Q(P(k_loop_var + 1))
                # next_config_list will store counts for Q(P(0))...Q(P(k_loop_var+1))
                next_config_list = list(prev_config_tuple) 
                next_config_list[next_Q_idx] += 1
                next_config_tuple = tuple(next_config_list)
                
                # Update dp_next
                state_key = (actual_good_count_for_dp, next_P_idx, next_config_tuple)
                dp_next[state_key] = (dp_next.get(state_key, 0) + ways) % MOD
        
        dp_curr = dp_next

    # Sum ways for all states meeting "at least K_limit" condition
    ans = 0
    for state_tuple, ways in dp_curr.items():
        good_count, _, _ = state_tuple
        # Due to capping, states with >= K_limit good strings are stored with good_count == K_limit
        if good_count == K_limit: 
            ans = (ans + ways) % MOD
            
    sys.stdout.write(str(ans) + "
")

if __name__ == '__main__':
    main()