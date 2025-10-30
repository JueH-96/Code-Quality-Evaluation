import sys

# Set recursion limit for potentially deep calls, though not strictly needed here
# sys.setrecursionlimit(10**6) 

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353
    INV2 = pow(2, MOD - 2, MOD) # Precompute inverse of 2 for IFFWHT and calculations

    # MAX_VAL determines the size of the FWHT domain.
    # A_i < 2^20, so 2^20 (1 << 20) is the appropriate MAX_VAL.
    MAX_VAL = 1 << 20 # 2^20 = 1048576

    # --- FWHT related functions ---
    # Performs FWHT (XOR basis) on the input array.
    # If inverse is True, performs inverse FWHT.
    def fwht(arr, inverse):
        h = 1
        while h < MAX_VAL:
            for i in range(0, MAX_VAL, h * 2):
                for j in range(h):
                    u = arr[i + j]
                    v = arr[i + j + h]
                    arr[i + j] = (u + v) % MOD
                    arr[i + j + h] = (u - v + MOD) % MOD # Ensure result is positive
            h *= 2

        if inverse:
            inv_max_val = pow(MAX_VAL, MOD - 2, MOD) # Precompute inverse of MAX_VAL
            for i in range(MAX_VAL):
                arr[i] = (arr[i] * inv_max_val) % MOD

    # --- Dynamic Programming with FWHT ---
    # dp_f[j][k] stores the k-th component in the FWHT domain
    # for subsequences with length `j` modulo `M`.
    dp_f = [[0] * MAX_VAL for _ in range(M)]

    # Initial state: only the empty subsequence
    # It has length 0, XOR sum 0. Counts: [1, 0, 0, ...]
    # FWHT([1, 0, ...]) results in [1, 1, ...]
    dp_f[0] = [1] * MAX_VAL

    # Process each element A_val from the input sequence A
    for A_val in A:
        # H_A_val[k] = (-1)^popcount(A_val & k)
        # This acts as the "multiplier" for the XOR shift in FWHT domain.
        H_A_val = [0] * MAX_VAL
        for k_idx in range(MAX_VAL):
            # Calculate popcount for (A_val & k_idx)
            # Python's int.bit_count() is efficient for this.
            if (A_val & k_idx).bit_count() % 2 == 0:
                H_A_val[k_idx] = 1
            else:
                H_A_val[k_idx] = MOD - 1 # Equivalent to -1 mod MOD

        # To correctly update dp_f, we need to use values from the previous iteration.
        # A deep copy ensures we're using the state *before* processing the current A_val.
        prev_dp_f = [list(row) for row in dp_f]
        
        # Update dp_f for all k values based on current A_val
        for k_idx in range(MAX_VAL):
            current_H = H_A_val[k_idx] # H_A_val[k_idx] is the factor for this k-th component
            
            # Iterate through all modulo M length states (j)
            for j in range(M):
                # The term contributing to length j from length (j-1)mod M
                prev_j = (j - 1 + M) % M 
                term_from_prev_len = (prev_dp_f[prev_j][k_idx] * current_H) % MOD
                
                # The total count for F[j][k] is sum of:
                # 1. Existing subsequences of length j (not including current A_val)
                # 2. Subsequences of length (j-1+M)%M that gain length by adding A_val
                dp_f[j][k_idx] = (prev_dp_f[j][k_idx] + term_from_prev_len) % MOD
        
    # --- Final calculation of total score ---
    # We are interested in subsequences whose length is a multiple of M.
    # These are stored in dp_f[0].
    
    # Convert dp_f[0] from FWHT domain back to actual counts.
    # After this call, dp_f[0][x] will be the number of subsequences with XOR sum `x`
    # and length being a multiple of `M`.
    fwht(dp_f[0], True) 

    total_score = 0
    for x in range(MAX_VAL):
        count_x = dp_f[0][x]
        
        # Only process if there are such subsequences
        if count_x > 0:
            # Score is (XOR_sum)^K
            # The empty subsequence (XOR sum 0, length 0) is included in dp_f[0][0].
            # Its score is 0^K. Since K >= 1, 0^K = 0.
            # So, including it does not affect the sum of scores for non-empty subsequences.
            
            score_x = pow(x, K, MOD) # Compute x^K modulo MOD
            total_score = (total_score + count_x * score_x) % MOD

    # Print the final result
    sys.stdout.write(str(total_score) + "
")

# Call the solve function to run the program
if __name__ == "__main__":
    solve()