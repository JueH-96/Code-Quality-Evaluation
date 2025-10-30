# YOUR CODE HERE
import sys

# Standard input/output for competitive programming
# Using sys.stdin.readline for faster input
input = sys.stdin.readline

def solve():
    K = int(input())
    C = list(map(int, input().split()))

    MOD = 998244353

    # Precompute factorials and inverse factorials up to K
    # Max value needed for factorial is K (for L!) or max(C_i) (for 1/k!).
    # Since both K and max(C_i) can be up to 1000, K is sufficient for MAX_VAL_FACTORIAL.
    MAX_VAL_FACTORIAL = K 
    
    fact = [1] * (MAX_VAL_FACTORIAL + 1)
    inv_fact = [1] * (MAX_VAL_FACTORIAL + 1)

    # Calculate factorials
    for i in range(1, MAX_VAL_FACTORIAL + 1):
        fact[i] = (fact[i-1] * i) % MOD
    
    # Calculate inverse factorials using Fermat's Little Theorem (a^(MOD-2) % MOD)
    # MOD is a prime number, so this is valid.
    inv_fact[MAX_VAL_FACTORIAL] = pow(fact[MAX_VAL_FACTORIAL], MOD - 2, MOD)
    
    # Calculate inverse factorials for i < MAX_VAL_FACTORIAL using inv_fact[i] = inv_fact[i+1] * (i+1)
    for i in range(MAX_VAL_FACTORIAL - 1, -1, -1):
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    # dp[j] will store the coefficient of x^j in the product of generating functions
    # for the characters processed so far. Specifically, it's sum_{counts for length j} (1 / product(count_i!))
    
    # Initial state: dp[0] = 1 (representing the empty string, length 0, coefficient 1)
    # All other dp[j] are 0 initially.
    dp = [0] * (K + 1)
    dp[0] = 1

    # Iterate through each character type (A-Z)
    for char_idx in range(26):
        c_limit = C[char_idx] # Maximum allowed occurrences for the current character
        
        # Create a temporary array for the new DP state after considering the current character.
        # This prevents using values from the current character in calculations for itself within the same char_idx loop.
        new_dp = [0] * (K + 1)
        
        # Iterate over all possible lengths 'current_len' accumulated from previous characters
        for current_len in range(K + 1):
            # If dp[current_len] is 0, no strings of this length can be formed with previous characters,
            # so no new strings can be formed by adding current character.
            if dp[current_len] == 0:
                continue
            
            # Iterate over 'k', the number of occurrences of the current character we add
            # 'k' ranges from 0 up to its allowed limit (c_limit)
            for k in range(c_limit + 1):
                # Calculate the total length if 'k' instances of the current character are added
                new_len = current_len + k
                
                # If the new length exceeds K, we stop adding more of the current character.
                # Since k is increasing, all subsequent new_len for this current_len will also be > K.
                if new_len > K:
                    break 
                
                # The contribution to new_dp[new_len] from dp[current_len] is:
                # dp[current_len] * (1/k!)
                # This corresponds to multiplying the polynomial representing previous characters (dp)
                # by the polynomial for the current character (sum x^k/k!).
                term = (dp[current_len] * inv_fact[k]) % MOD
                new_dp[new_len] = (new_dp[new_len] + term) % MOD
        
        # Update dp to new_dp for the next character type.
        # This makes new_dp the current dp state for the next iteration.
        dp = new_dp

    # After iterating through all 26 characters, dp[L] contains the sum of 1/(x_1! * ... * x_26!)
    # for all valid count combinations (x_1, ..., x_26) that sum to L.
    # The actual number of distinct strings of length L for a given multiset of characters
    # (x_1, ..., x_26) is L! / (x_1! * ... * x_26!).
    # So, the total number of strings of length L is dp[L] * L!.
    
    total_answer = 0
    # Sum up the counts for strings of length from 1 to K (inclusive)
    for L in range(1, K + 1):
        num_strings_len_L = (dp[L] * fact[L]) % MOD
        total_answer = (total_answer + num_strings_len_L) % MOD

    print(total_answer)

# Call the solve function to execute the program
solve()