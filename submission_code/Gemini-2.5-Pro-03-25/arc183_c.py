import sys

# Set higher recursion depth if needed, although the solution is iterative.
# sys.setrecursionlimit(2000) 

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    conditions = []
    for _ in range(M):
        conditions.append(list(map(int, sys.stdin.readline().split())))

    MOD = 998244353

    # Precompute factorials and their modular inverses
    # Factorials up to N
    fact = [1] * (N + 1)
    # Modular inverses of factorials up to N
    inv_fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i-1] * i) % MOD

    # Compute modular inverse of N! using Fermat's Little Theorem: a^(p-2) mod p
    # Works because MOD is prime.
    inv_fact[N] = pow(fact[N], MOD - 2, MOD)
    # Compute modular inverses of factorials iteratively downwards
    for i in range(N - 1, -1, -1):
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    # Function to compute nCr mod P using precomputed values
    def nCr_mod(n, r):
        # Basic checks for invalid r
        if r < 0 or r > n:
            return 0
        # Base cases C(n, 0) = C(n, n) = 1
        if r == 0 or r == n:
             return 1
        # Optimization: Use symmetry C(n, k) = C(n, n-k)
        if r > n // 2: 
             r = n - r
        
        # Calculate nCr = n! / (r! * (n-r)!) using modular arithmetic
        # n! * (r!)^{-1} * ((n-r)! )^{-1} mod P
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n-r]) % MOD
        return (num * den) % MOD

    # Precompute IsForbidden[k][l][r] information.
    # `is_forbidden[idx]` stores 1 if position k is forbidden as max for interval [l, r], 0 otherwise.
    # Uses 0-based indexing for k, l, r internally.
    # Uses a flattened bytearray for memory efficiency: size N * N * N bytes. 
    # For N=500, this is 125MB, which is acceptable.
    is_forbidden = bytearray(N * N * N)

    # Group conditions by X_i to process all conditions related to a fixed X_i=k together.
    conditions_by_X = [[] for _ in range(N)]
    for i in range(M):
        # Read condition L_i, R_i, X_i (1-based)
        L, R, X = conditions[i]
        # Store using 0-based indices for internal calculations
        conditions_by_X[X-1].append((L-1, R-1))

    # Precomputation step using 2D difference array technique for each k
    for k in range(N): # k is the 0-based index of the position being considered for maximum
        # Skip if there are no conditions involving X_i = k+1
        if not conditions_by_X[k]:
            continue

        # Initialize a 2D array `Mark` for the difference array calculation.
        # Size (N+2) x (N+2) allows using 1-based indexing for grid operations conveniently.
        Mark = [[0] * (N + 2) for _ in range(N + 2)]

        # Process each condition (L_i, R_i) associated with the current k (originally X_i = k+1)
        for L_i, R_i in conditions_by_X[k]: # L_i, R_i are 0-based interval endpoints
             # A state (l, r) in DP means we are assigning values to positions l..r.
             # Placing the maximum value at position k within [l, r] is forbidden if
             # there exists a condition (L_i, R_i, X_i=k+1) such that [L_i, R_i] is contained within [l, r].
             # This translates to: k must be forbidden for any DP state (l, r) where l <= L_i and r >= R_i.
             # We mark this region in the (l, r) plane using the difference array.
             
             # Convert the region l in [0, L_i], r in [R_i, N-1] to 1-based grid coordinates.
             # The row index corresponds to l, column index corresponds to r.
             # The region becomes: row index in [1, L_i+1], column index in [R_i+1, N]
             l1, r1 = 1, R_i + 1 
             l2, r2 = L_i + 1, N 
             
             # Apply the 2D difference array update only if the region is valid
             # (prevents issues with indices if L_i or R_i are at boundaries)
             if l1 <= l2 and r1 <= r2: 
                 Mark[l1][r1] += 1
                 Mark[l1][r2 + 1] -= 1
                 Mark[l2 + 1][r1] -= 1
                 Mark[l2 + 1][r2 + 1] += 1

        # Compute prefix sums on the Mark grid to get the final count for each cell (l, r)
        # This count indicates how many conditions make k forbidden for state (l, r).
        for l_idx in range(1, N + 2): # Iterate through rows (corresponds to l)
            for r_idx in range(1, N + 2): # Iterate through columns (corresponds to r)
                 # Standard 2D prefix sum calculation
                 Mark[l_idx][r_idx] += Mark[l_idx-1][r_idx] + Mark[l_idx][r_idx-1] - Mark[l_idx-1][r_idx-1]

        # Store the precomputed forbidden status into the flattened `is_forbidden` array
        # Map the 1-based grid coordinates back to 0-based DP state indices (l, r)
        for l in range(N): # l is 0-based start index of DP interval
            for r in range(l, N): # r is 0-based end index of DP interval
                 # The value Mark[l+1][r+1] corresponds to the state (l, r).
                 # If this count is > 0, it means k is a forbidden position for interval [l, r].
                 if Mark[l+1][r+1] > 0:
                     # Calculate the flat index for state (k, l, r)
                     idx = k*N*N + l*N + r
                     # Basic bounds check for safety
                     if idx < len(is_forbidden): 
                         is_forbidden[idx] = 1 # Mark this state as forbidden (byte value 1)

    # Dynamic Programming calculation
    # dp[l][r] stores the number of valid relative permutations for interval [l, r] (0-based)
    dp = [[0] * N for _ in range(N)]

    # Base case: Intervals of length 1 (dp[i][i])
    for i in range(N):
       # Calculate flat index for state (k=i, l=i, r=i)
       idx = i*N*N + i*N + i
       # Position i is the only choice for maximum in interval [i, i].
       # Check if this position is forbidden.
       if idx < len(is_forbidden) and not is_forbidden[idx]:
           dp[i][i] = 1 # If not forbidden, there is 1 way (place the single element)
       # else dp[i][i] remains 0 (initialized value)

    # Calculate DP table iteratively based on the length of the interval
    for length in range(2, N + 1):
        # Iterate through all possible start positions `l` for an interval of current length
        for l in range(N - length + 1):
            r = l + length - 1 # Calculate end position `r`
            
            current_dp_val = 0 # Accumulator for dp[l][r]
            
            # Iterate through all possible positions `k` within [l, r] for the maximum element
            for k in range(l, r + 1):
                # Check if placing the maximum element at position `k` is forbidden for interval [l, r]
                # Calculate the flat index for state (k, l, r)
                idx = k*N*N + l*N + r
                
                # If position k is NOT forbidden for this interval [l, r]
                if idx < len(is_forbidden) and not is_forbidden[idx]:
                    
                    # Get the number of ways for the left subproblem: interval [l, k-1]
                    # If k=l, the left interval is empty, number of ways is 1.
                    ways_left = dp[l][k-1] if k > l else 1
                    
                    # Get the number of ways for the right subproblem: interval [k+1, r]
                    # If k=r, the right interval is empty, number of ways is 1.
                    ways_right = dp[k+1][r] if k < r else 1
                    
                    # Calculate combinations C(length-1, k-l)
                    # The maximum element is placed at k. 
                    # The remaining 'length-1' elements must be assigned to the remaining positions.
                    # We choose 'k-l' elements out of 'length-1' to go to the left interval [l, k-1].
                    # The values assigned are relative {1, ..., length}. Max value is 'length'.
                    comb = nCr_mod(length - 1, k - l)
                    
                    # Combine results using multiplication modulo MOD
                    term = (comb * ways_left) % MOD
                    term = (term * ways_right) % MOD
                    
                    # Add the contribution of this `k` to the total ways for dp[l][r]
                    current_dp_val = (current_dp_val + term) % MOD
            
            # Store the computed value for dp[l][r]
            dp[l][r] = current_dp_val

    # The final answer is the number of ways for the full interval [0, N-1]
    print(dp[0][N-1])

solve()