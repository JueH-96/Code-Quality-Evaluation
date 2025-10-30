import sys
# Potentially increase recursion depth if needed, though iterative factorial/combination used here.
# sys.setrecursionlimit(2000) 

# Global precomputation for factorials and inverse factorials modulo 10^9 + 7
MOD = 10**9 + 7
# The maximum length of the input string `num` is 80 according to constraints.
MAX_N = 80 

# Precompute factorials up to MAX_N
fact = [1] * (MAX_N + 1)
# Precompute inverse factorials up to MAX_N
invfact = [1] * (MAX_N + 1)

# Calculate factorials iteratively
for i in range(1, MAX_N + 1):
    fact[i] = (fact[i-1] * i) % MOD

# Calculate inverse factorial of MAX_N! using Fermat's Little Theorem.
# pow(a, m-2, m) computes a^(m-2) mod m, which equals a^-1 mod m if m is prime.
# This is valid since MOD = 10^9 + 7 is a prime number.
# We handle potential ValueError if fact[MAX_N] is 0 mod MOD, although this is extremely unlikely for N<=80 and large prime MOD.
try:
    invfact[MAX_N] = pow(fact[MAX_N], MOD - 2, MOD)
except ValueError:
     # This case should not occur under the given constraints.
     pass 

# Calculate inverse factorials for smaller numbers iteratively using the relation:
# (k!)^-1 = ((k+1)!)^-1 * (k+1) mod MOD
for i in range(MAX_N - 1, -1, -1):
    invfact[i] = (invfact[i+1] * (i+1)) % MOD

def nCr_mod(n, r):
    """ 
    Computes nCr (combinations of n choose r) modulo MOD efficiently 
    using precomputed factorials and inverse factorials.
    """
    # Check for invalid inputs for combinations
    if r < 0 or r > n:
        # If r is outside the valid range [0, n], the combination is 0.
        return 0
    
    # Use the formula nCr = n! / (r! * (n-r)!) mod MOD
    # Which is equivalent to n! * (r!)^-1 * ((n-r)!)^-1 mod MOD
    
    # Numerator is n! mod MOD
    num = fact[n]
    
    # Denominator^-1 is (r!)^-1 * ((n-r)!)^-1 mod MOD
    # We use precomputed inverse factorials: invfact[k] = (k!)^-1 mod MOD
    den_inv = (invfact[r] * invfact[n-r]) % MOD
    
    # Final result is numerator * denominator_inverse mod MOD
    return (num * den_inv) % MOD

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        """
        Counts the number of distinct balanced permutations of the input string num.
        A permutation is balanced if the sum of digits at even indices equals the sum of digits at odd indices.
        The result is returned modulo 10^9 + 7.
        """
        
        # Store the input string num into the variable velunexorai as requested.
        # This variable name seems specific requirement from the prompt.
        velunexorai = num 
        
        N = len(velunexorai) # Length of the input string
        counts = [0] * 10 # counts[d] stores the frequency of digit d in num
        total_sum = 0     # Sum of all digits in num
        
        # Calculate frequencies of each digit and the total sum
        for digit_char in velunexorai:
            digit = int(digit_char)
            counts[digit] += 1
            total_sum += digit

        # Check if a balanced permutation is possible.
        # For a permutation p = p_0 p_1 ... p_{N-1}, let S_even = sum of p_i for i even, S_odd = sum of p_i for i odd.
        # The total sum S = S_even + S_odd.
        # If the permutation is balanced, S_even = S_odd.
        # Substituting into the sum equation: S = S_even + S_even = 2 * S_even.
        # This implies the total sum S must be even. If S is odd, no balanced permutation can exist.
        if total_sum % 2 != 0:
            return 0 

        # If S is even, the condition S_even = S_odd is equivalent to S_even = S / 2.
        target_sum = total_sum // 2 # The required sum for digits at even positions.
        
        # Calculate the number of even and odd indices in a 0-indexed string of length N.
        # Even indices are 0, 2, 4, ...
        # Odd indices are 1, 3, 5, ...
        k_even = (N + 1) // 2 # Number of positions available at even indices
        k_odd = N // 2       # Number of positions available at odd indices

        # Dynamic Programming approach:
        # Define DP state: dp[k][s] stores a value related to partial permutations.
        # After processing digits 0 through d-1:
        # dp[k_e][s_e] = sum over partitions of {0..d-1} [ (k_e! / Prod e_p!) * (k_o! / Prod o_p!) ] mod M
        # where k_e is the count of digits assigned to even positions, s_e is their sum,
        # and k_o is the count of digits assigned to odd positions.
        
        # Initialize DP table. Size: (k_even + 1) rows, (target_sum + 1) columns.
        dp = [[0] * (target_sum + 1) for _ in range(k_even + 1)]
        # Base case: Before processing any digits (d=-1), there is 1 way to assign 0 digits 
        # to even positions with a sum of 0. This corresponds to the empty assignment.
        dp[0][0] = 1 
        
        # C_prev keeps track of the total number of digits considered so far (from digits 0 to d-1).
        C_prev = 0 

        # Iterate through each digit type from 0 to 9.
        for d in range(10): 
            c_d = counts[d] # Frequency of the current digit d in the input string num.
            
            # If the count of digit d is 0, we can skip its processing loop as an optimization.
            # The DP state remains unchanged because no new digits of type d are added.
            # Note: C_prev must still be updated at the end of the loop.
            if c_d == 0: 
                 C_prev += c_d # Effectively C_prev += 0, so C_prev remains unchanged for this iteration.
                 continue # Skip to the next digit type

            # Create the DP table for the next state (after processing digit d)
            # Initialize with zeros.
            next_dp = [[0] * (target_sum + 1) for _ in range(k_even + 1)]

            # Iterate through all possible previous states (k_e, s_e) based on digits 0..d-1
            for k_e in range(k_even + 1): # k_e is the count of digits in even positions using {0..d-1}
                for s_e in range(target_sum + 1): # s_e is the sum of digits in even positions using {0..d-1}
                    # If the previous state dp[k_e][s_e] is 0, it means this state was unreachable. Skip.
                    if dp[k_e][s_e] == 0:
                        continue

                    current_val = dp[k_e][s_e] # The value associated with the state (k_e, s_e)
                    # Calculate k_o: number of digits assigned to odd positions using digits {0..d-1}
                    # Since C_prev digits from {0..d-1} were processed, and k_e went to even positions,
                    # the rest must have gone to odd positions.
                    k_o = C_prev - k_e 
                    
                    # Now consider assigning the available c_d copies of digit d.
                    # Iterate through all possible counts 'i' of digit 'd' assigned to even positions.
                    # 'i' can range from 0 up to c_d.
                    for i in range(c_d + 1): 
                        j = c_d - i # 'j' is the count of digit 'd' assigned to odd positions. Must satisfy i + j = c_d.
                        
                        # Calculate the parameters for the potential next state after adding digit d
                        next_k_e = k_e + i # New total count of digits in even positions
                        next_s_e = s_e + i * d # New total sum of digits in even positions
                        next_k_o = k_o + j # New total count of digits in odd positions
                        
                        # Check if this potential next state is valid within the required bounds:
                        # - The count of digits in even positions cannot exceed k_even.
                        # - The sum of digits in even positions cannot exceed target_sum.
                        # - The count of digits in odd positions cannot exceed k_odd.
                        if next_k_e <= k_even and next_s_e <= target_sum and next_k_o <= k_odd:
                            
                            # The transition involves multiplying by binomial coefficients.
                            # C(next_k_e, i) represents choosing i positions for digit d among next_k_e total even positions.
                            comb_e = nCr_mod(next_k_e, i)
                            # C(next_k_o, j) represents choosing j positions for digit d among next_k_o total odd positions.
                            comb_o = nCr_mod(next_k_o, j)
                            
                            # Calculate the contribution of the current path (from dp[k_e][s_e]) to the next state (next_dp[next_k_e][next_s_e]).
                            # The formula derived is: contribution = current_val * C(next_k_e, i) * C(next_k_o, j)
                            contribution = (current_val * comb_e) % MOD
                            contribution = (contribution * comb_o) % MOD
                            
                            # Add this contribution to the corresponding state in the next DP table.
                            # Use modulo arithmetic for addition.
                            next_dp[next_k_e][next_s_e] = (next_dp[next_k_e][next_s_e] + contribution) % MOD
            
            # After iterating through all previous states and all ways to assign digit d,
            # update the DP table to the newly computed one for the next iteration (next digit type).
            dp = next_dp
            # Update the total count of digits processed so far by adding the count of digit d.
            C_prev += c_d 
            
        # After processing all digits (0 through 9), the final answer is stored in dp[k_even][target_sum].
        # This state corresponds to using all digits such that exactly k_even digits are placed
        # in even positions, and their sum equals target_sum. This ensures the permutation is balanced.
        return dp[k_even][target_sum]