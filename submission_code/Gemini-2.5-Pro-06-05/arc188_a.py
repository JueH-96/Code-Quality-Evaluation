import sys
from collections import defaultdict

def solve():
    """
    Reads input, runs the DP algorithm, and prints the answer.
    """
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    MOD = 998244353

    # State mapping for s_i = ((ca_i-cb_i)%2, (cb_i-cc_i)%2)
    s_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    s_map_rev = {v: k for k, v in s_map.items()}
    
    # Change in state vector for each character
    delta_map = {'A': (1, 0), 'B': (1, 1), 'C': (0, 1)}

    # DP with sparse states using a dictionary
    # dp_curr maps state_tuple -> ways
    # state_tuple is (n0, n1, n2, s_idx)
    # n0, n1, n2: counts of states 0, 1, 2 among s_0, ..., s_i
    # s_idx: index of state s_i
    dp_curr = defaultdict(int)
    
    # Base case: i=0 (empty prefix).
    # The state is s_0 = (0,0) (index 0).
    # The sequence of states is {s_0}, so n0=1.
    dp_curr[(1, 0, 0, 0)] = 1

    for i in range(N):
        dp_next = defaultdict(int)
        char_options = ['A', 'B', 'C'] if S[i] == '?' else [S[i]]
        
        for state, ways in dp_curr.items():
            if ways == 0:
                continue
            
            n0, n1, n2, s_idx = state
            s_curr_tuple = s_map_rev[s_idx]
            
            for char in char_options:
                delta = delta_map[char]
                s_next_tuple = ((s_curr_tuple[0] + delta[0]) % 2, 
                                (s_curr_tuple[1] + delta[1]) % 2)
                s_next_idx = s_map[s_next_tuple]
                
                n0_next, n1_next, n2_next = n0, n1, n2
                if s_next_idx == 0:
                    n0_next += 1
                elif s_next_idx == 1:
                    n1_next += 1
                elif s_next_idx == 2:
                    n2_next += 1
                
                next_state = (n0_next, n1_next, n2_next, s_next_idx)
                dp_next[next_state] = (dp_next[next_state] + ways) % MOD
        
        dp_curr = dp_next

    total_ans = 0
    
    # After N steps, dp_curr holds results for the full string (N+1 prefixes s_0..s_N)
    for state, ways in dp_curr.items():
        n0, n1, n2, _ = state
        n3 = (N + 1) - (n0 + n1 + n2)
        
        good_substring_count = 0
        good_substring_count += n0 * (n0 - 1) // 2
        good_substring_count += n1 * (n1 - 1) // 2
        good_substring_count += n2 * (n2 - 1) // 2
        good_substring_count += n3 * (n3 - 1) // 2
        
        if good_substring_count >= K:
            total_ans = (total_ans + ways) % MOD
            
    print(total_ans)

solve()