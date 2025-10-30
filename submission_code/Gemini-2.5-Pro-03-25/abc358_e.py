# YOUR CODE HERE
import sys

def solve():
    # Read the maximum length K from input
    K = int(sys.stdin.readline())
    # Read the constraints C_1, C_2, ..., C_26 from input
    C = list(map(int, sys.stdin.readline().split()))
    
    # Define the modulus for calculations
    MOD = 998244353

    # --- Precomputation ---
    # The problem constraints state 1 <= K <= 1000.
    
    # Precompute factorials modulo MOD up to K
    # fact[i] will store i! mod MOD
    fact = [1] * (K + 1)
    for i in range(1, K + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    # Precompute inverse factorials modulo MOD up to K
    # invfact[i] will store (i!)^(-1) mod MOD
    invfact = [1] * (K + 1)
    # Compute the modular inverse of K! using Fermat's Little Theorem:
    # a^(p-2) mod p is the modular inverse of a modulo p, for prime p.
    invfact[K] = pow(fact[K], MOD - 2, MOD)
    # Compute inverse factorials for smaller numbers efficiently using the relation:
    # (i!)^{-1} = ((i+1)! * (i+1)^{-1})^{-1} = (i+1)!^{-1} * (i+1) mod P
    # So, invfact[i] = invfact[i+1] * (i+1) % MOD
    for i in range(K - 1, -1, -1):
        invfact[i] = (invfact[i + 1] * (i + 1)) % MOD

    # Define a function to compute combinations C(n, r) modulo MOD
    # Uses the precomputed factorials and inverse factorials
    def nCr_mod(n, r):
        # Handle invalid arguments where r < 0 or r > n
        if r < 0 or r > n:
            return 0
        # Base cases: C(n, 0) = 1, C(n, n) = 1
        if r == 0 or r == n:
             return 1
        # Use the symmetry property C(n, r) = C(n, n-r)
        # This can potentially make calculations faster if r > n/2 by reducing r.
        if r > n // 2:
            r = n - r
            
        # Calculate nCr = n! / (r! * (n-r)!) mod MOD
        # Use modular arithmetic: nCr = n! * (r!)^{-1} * ((n-r)! )^{-1} mod MOD
        num = fact[n]
        # Compute the modular inverse of the denominator (r! * (n-r)!)
        den_inv = (invfact[r] * invfact[n - r]) % MOD
        # Final result is (numerator * denominator_inverse) mod MOD
        return (num * den_inv) % MOD

    # --- Dynamic Programming ---
    # Initialize DP table. dp[j] will store the number of valid strings of length j
    # considering the letters processed so far.
    dp = [0] * (K + 1)
    # Base case: There is one string of length 0 (the empty string) which is always valid initially.
    dp[0] = 1 

    # Iterate through each of the 26 uppercase English letters (indexed 0 to 25)
    for i in range(26):
        # Get the maximum allowed count for the current letter (a_{i+1})
        Ci = C[i] 
        
        # Create a temporary DP table for the next state.
        # This avoids modifying the current `dp` state while it's still needed for calculations.
        next_dp = [0] * (K + 1)
        
        # Compute the next state `next_dp` based on the current state `dp`.
        # Iterate through all possible lengths `j` of strings formed using letters processed *before* the current one.
        for j in range(K + 1): 
            # Optimization: If `dp[j]` is 0, it means no strings of length `j` were possible with the previous set of letters.
            # These cannot contribute to forming longer strings by adding the current letter. Skip.
            if dp[j] == 0: 
                continue
                
            # Consider adding `k` instances of the current letter (letter with index `i`).
            # The count `k` must be between 0 and `Ci` (inclusive) according to the problem constraints.
            for k in range(Ci + 1): 
                # Calculate the new length of the string after adding `k` letters.
                new_len = j + k 
                
                # If the new length exceeds the maximum allowed length K, then any larger `k` for the current `j` will also result in `new_len > K`.
                # So we can stop iterating through `k` for this `j` (break from the inner loop).
                if new_len > K: 
                    break 
                
                # Calculate the number of ways to choose `k` positions for the new letter among the `new_len` total positions.
                # This is given by the binomial coefficient C(new_len, k).
                comb = nCr_mod(new_len, k)
                
                # The number of ways to form a string of length `new_len` using letters up to index `i`,
                # by taking a string of length `j` (using letters up to `i-1`) and inserting `k` copies of letter `i`,
                # is `dp[j]` (number of ways to form the length `j` string) * `comb` (number of ways to place the `k` new letters).
                term = (dp[j] * comb) % MOD
                
                # Add this contribution to the total count for strings of length `new_len` in the `next_dp` state.
                # Use modular addition.
                next_dp[new_len] = (next_dp[new_len] + term) % MOD
                
        # Update the DP state: the computed `next_dp` becomes the current `dp` state for the next letter's iteration.
        dp = next_dp 

    # --- Final Answer Calculation ---
    # After processing all 26 letters, the `dp` array contains the counts of valid strings for lengths 0 to K.
    # The problem asks for the total number of strings with length between 1 and K, inclusive.
    # Sum up the counts `dp[L]` for `L` from 1 to K.
    total_count = 0
    for L in range(1, K + 1):
        total_count = (total_count + dp[L]) % MOD

    # Print the final total count modulo MOD.
    print(total_count)

# Execute the main function to solve the problem based on standard input.
solve()