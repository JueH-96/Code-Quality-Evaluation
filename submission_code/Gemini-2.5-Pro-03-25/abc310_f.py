# YOUR CODE HERE
import sys

# Main execution function
def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    MOD = 998244353

    # Modular exponentiation function (standard implementation)
    # Computes (base^exp) % modulus
    def modpow(base, exp, modulus):
        base %= modulus
        result = 1
        while exp > 0:
            if exp % 2 == 1: result = (result * base) % modulus
            base = (base * base) % modulus
            exp //= 2
        return result

    # Modular inverse function using Fermat's Little Theorem
    # Computes n^(-1) % modulus where modulus is prime
    # This is equivalent to n^(modulus-2) % modulus
    def modInverse(n, modulus):
        # Compute n^(modulus-2) % modulus using modpow
        return modpow(n, modulus - 2, modulus)

    # Initialize DP table using a dictionary. Key is mask, value is count.
    # dp[mask] stores the number of ways to assign outcomes to the dice processed so far
    # such that NO subset of these outcomes sums to 10.
    # 'mask' is a bitmask representing the set of achievable subset sums k in the range [1, 9].
    # Specifically, the (k-1)-th bit (0-indexed) is set if sum k is achievable.
    dp = {0: 1} # Base case: With 0 dice, the only achievable sum is 0 (empty set). 
                # This state is represented by mask 0. There is 1 way to achieve this (do nothing).
    
    # Process each die one by one
    for i in range(N):
        Ai = A[i] # Number of faces for the current die (die i+1)
        
        # DP table for the next step (after considering die i)
        new_dp = {} 

        # Iterate through states reachable after processing the first i dice
        for mask, count in dp.items():
            # If count is 0 for this state, it means this state wasn't reachable, so we skip it.
            if count == 0: continue
            
            # Determine the set K of achievable sums (1 to 9) based on the current mask
            K = []
            for k in range(1, 10):
                if (mask >> (k-1)) & 1: # Check if (k-1)-th bit is set
                    K.append(k)
            
            # Determine the set D of "forbidden" values for the current die's outcome (X_{i+1})
            # These are values that, if rolled, would immediately allow forming a subset sum of 10.
            D = {10} # Rolling 10 is forbidden because {10} itself sums to 10.
            for k in K:
                # If sum k is achievable, then rolling 10-k would make sum 10 possible (k + (10-k) = 10).
                # We only need to consider positive values for 10-k.
                if 10 - k >= 1:
                     D.add(10 - k) 

            # --- Calculate transitions based on the possible outcome x of the current die (X_{i+1}) ---
            
            # Part 1: Consider outcomes x > 10
            # Rolling a value x > 10 cannot help form a subset sum of 10 using sums less than 10.
            # Such outcomes do not change the set of achievable sums less than 10. The mask remains 'mask'.
            # Count how many such values are possible for the current die: values from 11 to Ai.
            # The number of such values is max(0, Ai - 10).
            count_gt_10 = max(0, Ai - 10) 
            if count_gt_10 > 0:
                 # Add the count contribution to the same state 'mask' in the next DP table.
                 current_val = new_dp.get(mask, 0)
                 # There are 'count' ways to reach 'mask' state, each can be extended by count_gt_10 ways.
                 new_dp[mask] = (current_val + count * count_gt_10) % MOD

            # Part 2: Consider outcomes x in [1, 9]
            # These outcomes might change the mask. Need to check if they are possible and not forbidden.
            for x in range(1, 10):
                # Check if outcome x is possible for this die (1 <= x <= Ai)
                if x <= Ai:
                    # Check if outcome x is forbidden (i.e., x is in set D)
                    if x not in D:
                        # If x is a possible outcome for this die and it's not forbidden:
                        # This path contributes to the count of "bad" outcomes (where 10 is not achievable).
                        
                        # Calculate the new mask resulting from adding outcome x.
                        new_mask = mask
                        # The value x itself becomes an achievable sum (if 1 <= x <= 9)
                        new_mask |= (1 << (x-1)) # Set bit x-1
                        
                        # Combine x with previously achievable sums k from set K
                        for k in K: # Iterate through sums k already achievable (1..9)
                            if k + x < 10: # If the new sum k+x is still less than 10
                                # This new sum k+x is now achievable. Set the corresponding bit.
                                new_mask |= (1 << (k + x - 1)) # Set bit (k+x)-1
                        
                        # Update the count for the resulting new_mask state in the next DP table.
                        current_val = new_dp.get(new_mask, 0)
                        # There are 'count' ways to reach the previous 'mask' state. Each leads to 'new_mask' with outcome x.
                        new_dp[new_mask] = (current_val + count) % MOD

        # The DP table for the next state replaces the current DP table for the next iteration.
        dp = new_dp

    # After processing all N dice, calculate the overall total number of possible outcomes.
    # This is the product of the number of faces for each die, modulo MOD.
    final_total_outcomes_product = 1
    for val in A:
        final_total_outcomes_product = (final_total_outcomes_product * val) % MOD

    # Sum the counts of all states in the final DP table.
    # This sum represents N_bad: the total number of outcomes where NO subset sums to 10.
    N_bad = 0
    for count in dp.values():
        N_bad = (N_bad + count) % MOD

    # Calculate N_good: the number of outcomes where at least one subset sums to 10.
    # N_good = Total Outcomes - N_bad. Perform calculation modulo MOD.
    # Add MOD before taking modulo to handle potential negative result from subtraction.
    N_good = (final_total_outcomes_product - N_bad + MOD) % MOD
    
    # The required probability is N_good / Total Outcomes.
    # Compute this using modular inverse: P = N_good * modInverse(Total Outcomes) % MOD.
    
    # If total_outcomes_product is 0 mod MOD (should not happen based on problem statement guarantee)
    # Handle division by zero defensively? Problem says denominator isn't divisible by MOD.
    if final_total_outcomes_product == 0:
         # This case implies impossible scenario given constraints or problem guarantee.
         # But if it somehow occurs, probability could be considered 0 or undefined.
         # Based on guarantees, this won't happen.
         pass

    total_outcomes_inv = modInverse(final_total_outcomes_product, MOD)
    probability = (N_good * total_outcomes_inv) % MOD
    
    # Print the final probability modulo MOD.
    print(probability)

# Execute the solve function when the script is run
solve()