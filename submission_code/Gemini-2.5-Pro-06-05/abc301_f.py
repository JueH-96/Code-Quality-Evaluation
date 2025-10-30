import sys
from collections import defaultdict

def main():
    """
    This function contains the main logic of the solution.
    It's wrapped in a function to avoid global scope issues.
    """
    S = sys.stdin.readline().strip()
    MOD = 998244353

    # Identify unique uppercase letters that appear as fixed characters in S.
    # These are "special" characters whose states we must track individually.
    special_uc_set = sorted(list(set(c for c in S if 'A' <= c <= 'Z')))
    m = len(special_uc_set)
    uc_map = {c: i for i, c in enumerate(special_uc_set)}
    
    # The rest of the uppercase letters are "anonymous" and can be treated symmetrically.
    num_anonymous_uc = 26 - m

    # DP state representation:
    # We use a dictionary `dp` where keys are states and values are the number of ways.
    # A state is a tuple: (s_tuple, k_tuple)
    # - s_tuple: A tuple of length `m`, where s_tuple[j] is the state of the j-th special uppercase char.
    # - k_tuple: A tuple (k1, k2, k3), where ki is the number of anonymous uppercase chars in state i.
    #
    # State definitions for an uppercase character C:
    # 0: Not seen yet.
    # 1: Seen once (subsequence "C" exists).
    # 2: Seen twice (subsequence "C, C" exists).
    # 3: "Armed" (subsequence "C, C, c" exists, where c is any lowercase letter).
    # If a path with an armed character encounters another uppercase letter, it forms a DDoS-type
    # subsequence, so that path becomes invalid.

    dp = defaultdict(int)
    
    # Initial state: All characters (special and anonymous) are in state 0.
    initial_s_tuple = tuple([0] * m)
    initial_k_tuple = (0, 0, 0)
    dp[(initial_s_tuple, initial_k_tuple)] = 1

    for char in S:
        next_dp = defaultdict(int)
        
        for state, count in dp.items():
            if count == 0:
                continue

            s_tuple, k_tuple = state
            k1, k2, k3 = k_tuple
            
            if char == '?':
                # --- Choice 1: Replace '?' with a lowercase letter (26 options) ---
                # All characters in state 2 become armed (state 3).
                next_s_list_low = list(s_tuple)
                for j in range(m):
                    if next_s_list_low[j] == 2:
                        next_s_list_low[j] = 3
                
                # Anonymous chars in state 2 also move to state 3.
                next_k_tuple_low = (k1, 0, k2 + k3)
                
                next_state_low = (tuple(next_s_list_low), next_k_tuple_low)
                next_dp[next_state_low] = (next_dp[next_state_low] + count * 26) % MOD

                # --- Choice 2 & 3: Replace '?' with an uppercase letter ---
                is_armed = any(s == 3 for s in s_tuple) or k3 > 0
                if not is_armed:
                    # 2a: Choose a special uppercase letter
                    for j in range(m):
                        s_j = s_tuple[j]
                        next_s_list_up = list(s_tuple)
                        if s_j == 0: next_s_list_up[j] = 1
                        elif s_j == 1: next_s_list_up[j] = 2
                        
                        next_state_up = (tuple(next_s_list_up), k_tuple)
                        next_dp[next_state_up] = (next_dp[next_state_up] + count) % MOD
                    
                    # 2b: Choose an anonymous uppercase letter
                    k0 = num_anonymous_uc - k1 - k2 - k3
                    if k0 > 0:
                        next_dp[(s_tuple, (k1 + 1, k2, k3))] = (next_dp[(s_tuple, (k1 + 1, k2, k3))] + count * k0) % MOD
                    if k1 > 0:
                        next_dp[(s_tuple, (k1 - 1, k2 + 1, k3))] = (next_dp[(s_tuple, (k1 - 1, k2 + 1, k3))] + count * k1) % MOD
                    if k2 > 0:
                        next_dp[(s_tuple, (k1, k2, k3))] = (next_dp[(s_tuple, (k1, k2, k3))] + count * k2) % MOD

            elif 'a' <= char <= 'z':
                # --- Fixed lowercase character ---
                next_s_list = list(s_tuple)
                for j in range(m):
                    if next_s_list[j] == 2:
                        next_s_list[j] = 3
                
                next_k_tuple = (k1, 0, k2 + k3)
                next_state = (tuple(next_s_list), next_k_tuple)
                next_dp[next_state] = (next_dp[next_state] + count) % MOD

            elif 'A' <= char <= 'Z':
                # --- Fixed uppercase character ---
                # An armed path encountering any uppercase character becomes invalid.
                is_armed = any(s == 3 for s in s_tuple) or k3 > 0
                if is_armed:
                    continue
                
                # If not armed, transition the state of the specific character `char`.
                j = uc_map[char]
                s_j = s_tuple[j]
                
                next_s_list = list(s_tuple)
                if s_j == 0: next_s_list[j] = 1
                elif s_j == 1: next_s_list[j] = 2
                
                next_state = (tuple(next_s_list), k_tuple)
                next_dp[next_state] = (next_dp[next_state] + count) % MOD

        dp = next_dp
    
    # The total number of valid strings is the sum of counts of all final states.
    total_count = sum(dp.values()) % MOD
    print(total_count)

if __name__ == "__main__":
    main()