MOD = 998244353

import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    n_k_line = data[0].split()
    N = int(n_k_line[0])
    K = int(n_k_line[1])
    S = data[1].strip()
    
    dp = {}
    init_state = (0, 1, 0, 0, 0)
    dp[init_state] = 1
    total_states_count = 1
    
    for idx in range(N):
        char = S[idx]
        choices = ['A', 'B', 'C'] if char == '?' else [char]
        new_dp = defaultdict(int)
        total_states_count += 1
        
        for state, count_val in dp.items():
            q, f0, f1, f2, g_val = state
            f3 = total_states_count - 1 - f0 - f1 - f2
            f_arr = [f0, f1, f2, f3]
            
            for ch in choices:
                if ch == 'A':
                    vec = (1, 0)
                elif ch == 'B':
                    vec = (1, 1)
                elif ch == 'C':
                    vec = (0, 1)
                else:
                    assert False, "Unexpected character"
                    
                x = (q >> 1) & 1
                y = q & 1
                nx = x ^ vec[0]
                ny = y ^ vec[1]
                new_q = (nx << 1) | ny
                
                count_j = f_arr[new_q]
                new_g = g_val + count_j
                if new_g > 1275:
                    new_g = 1275
                    
                new_f_arr = f_arr[:]
                new_f_arr[new_q] += 1
                new_state_tuple = (new_q, new_f_arr[0], new_f_arr[1], new_f_arr[2], new_g)
                new_dp[new_state_tuple] = (new_dp[new_state_tuple] + count_val) % MOD
                
        dp = new_dp

    total = 0
    for state, count_val in dp.items():
        q, f0, f1, f2, g_val = state
        if g_val >= K:
            total = (total + count_val) % MOD
            
    print(total)

if __name__ == '__main__':
    main()