import sys

def solve_problem():
    S_initial = sys.stdin.readline().strip()
    _ = int(sys.stdin.readline()) # Q is not directly used other than for loop count
    queries_K_str = sys.stdin.readline().split()
    queries_K = [int(k_str) for k_str in queries_K_str]

    L0 = len(S_initial)
    
    # memo = {} # Memoization can be added if performance issues arise, but usually not needed for this depth
    
    def get_char_recursive(k_idx, t_level):
        # key = (k_idx, t_level) # For memoization
        # if key in memo:
        #     return memo[key]

        if t_level == 0:
            # Base case: S_0. k_idx must be < L0 by construction of t_level.
            return S_initial[k_idx]

        len_S_prev_level = L0 * (1 << (t_level - 1))

        if k_idx < len_S_prev_level:
            result = get_char_recursive(k_idx, t_level - 1)
        else:
            char_from_S_prev = get_char_recursive(k_idx - len_S_prev_level, t_level - 1)
            result = char_from_S_prev.swapcase()
        
        # memo[key] = result # For memoization
        return result

    results = []
    for K_val in queries_K:
        k_query_idx = K_val - 1 # 0-indexed

        # Calculate effective_ops_level (t_eff)
        # t_eff is the smallest t such that k_query_idx < L0 * 2^t.
        # This is equivalent to (k_query_idx // L0).bit_length() for L0 > 0.
        # If k_query_idx < L0, then k_query_idx // L0 = 0. (0).bit_length() = 0. So t_eff = 0.
        # L0 >= 1 is guaranteed by constraints.
        effective_ops_level = (k_query_idx // L0).bit_length()
        
        ans_char = get_char_recursive(k_query_idx, effective_ops_level)
        results.append(ans_char)

    sys.stdout.write(" ".join(results) + "
")

solve_problem()