import sys

MOD = 998244353

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Precompute factorials and inverse factorials for combinations
    # Max catalan index needed is 2N-3.
    # C_k = binomial(2k, k) / (k+1). Max k is 2N-3.
    # So we need binomial(2*(2N-3), 2N-3) = binomial(4N-6, 2N-3).
    # Max value for n (in nCr(n,r)) is 4N.
    MAX_COMB = 4 * N 
    fact = [1] * (MAX_COMB + 1)
    inv_fact = [1] * (MAX_COMB + 1)
    for i in range(1, MAX_COMB + 1):
        fact[i] = (fact[i-1] * i) % MOD
    
    inv_fact[MAX_COMB] = pow(fact[MAX_COMB], MOD - 2, MOD)
    for i in range(MAX_COMB - 1, -1, -1):
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    def nCr_mod_p(n, r):
        if r < 0 or r > n:
            return 0
        return (((fact[n] * inv_fact[r]) % MOD) * inv_fact[n-r]) % MOD

    # Precompute Catalan numbers C_k = binomial(2k, k) / (k+1)
    # The max index for C_k needed is L + next_balance.
    # L = N-(idx+1). next_balance <= N-idx-1.
    # Max index is (N-(0+1)) + (N-0-1) = N-1 + N-1 = 2N-2.
    # So, need catalan up to C_{2N-2}. Max index is 2N-2.
    MAX_CATALAN_IDX = 2 * N
    catalan = [0] * (MAX_CATALAN_IDX + 1)
    for k in range(MAX_CATALAN_IDX + 1):
        val = nCr_mod_p(2 * k, k)
        val = (val * pow(k + 1, MOD - 2, MOD)) % MOD
        catalan[k] = val
    
    # Memoization table for DP: (idx, current_balance_offset, is_tight)
    # current_balance ranges from -1 (min) to N-idx-1 (max).
    # Smallest possible current_balance encountered overall is -1 (at idx=N).
    # Largest possible current_balance is N-1 (conceptually if B_0=N).
    # The range is roughly 0 to N. So offset by 1.
    # `current_balance + 1` ranges from 0 to N-idx.
    # `current_balance` for `s_k` cannot be less than -1.
    # So `current_balance + 1` maps to `0` for `current_balance = -1`.
    # Max `current_balance` (N-idx-1) for idx=0 is N-1. So max offset `N`.
    # So `current_balance_offset` from 0 to N. Max N+1 states for balance.
    # Total states `N * (N+1) * 2`. Max `3e5 * 3e5 * 2`.
    memo = {}

    # Set recursion limit for large N
    sys.setrecursionlimit(N + 100) 

    def solve_recursive(idx, current_balance, is_tight):
        # Base Case: All N elements placed
        if idx == N:
            return 1 if current_balance == -1 else 0 # Sequence is valid if final balance is -1
        
        # Memoization check (offset current_balance to be non-negative for dict key)
        cb_offset = current_balance + 1 
        if (idx, cb_offset, is_tight) in memo:
            return memo[(idx, cb_offset, is_tight)]

        res = 0
        # Determine the upper bound for the current digit B_idx
        upper_bound_d = A[idx] if is_tight else (N - 1) 

        for d in range(upper_bound_d + 1):
            next_balance = current_balance + d - 1
            
            # Check validity of this move based on balance rules
            is_valid_move = True
            
            # 1. Balance must not drop below 0 prematurely
            # Exception: if it's the last element (idx == N-1), balance can become -1.
            if next_balance < 0 and not (idx == N - 1 and next_balance == -1):
                is_valid_move = False
            
            # 2. Balance must not reach -1 prematurely (unless it's the very last step)
            if idx < N - 1 and next_balance == -1:
                is_valid_move = False
            
            if not is_valid_move:
                continue # Skip this `d` as it leads to an invalid sequence

            # Determine if the next state will still be 'tight'
            new_is_tight = is_tight and (d == upper_bound_d)
            
            if new_is_tight:
                # Still tight, recurse
                res = (res + solve_recursive(idx + 1, next_balance, True)) % MOD
            else:
                # Not tight: the remaining sequence B_idx...B_{N-1} can be formed freely (subject to B_i < N)
                # Use precomputed Catalan values for such suffixes.
                # Number of ways to complete a sequence of length L = N-(idx+1)
                # starting with `next_balance` and ending with `-1` (and staying non-negative)
                # is C_{L + next_balance}.
                
                L = N - (idx + 1)
                catalan_k_idx = L + next_balance
                
                count_suffix = 0
                if catalan_k_idx >= 0 and catalan_k_idx < len(catalan):
                    count_suffix = catalan[catalan_k_idx]
                
                res = (res + count_suffix) % MOD
        
        # Store result in memoization table and return
        memo[(idx, cb_offset, is_tight)] = res
        return res

    # Start the DP from the beginning of the sequence (idx=0), with initial balance 0, and in tight state.
    ans = solve_recursive(0, 0, True)
    sys.stdout.write(str(ans) + '
')