import math

class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        """
        Calculates the number of ways to express n as the sum of x-th powers 
        of unique positive integers, modulo 10^9 + 7.

        This problem can be solved using dynamic programming, similar to the 
        0/1 knapsack or subset sum count problem. We want to find the number 
        of ways to select a subset of unique numbers {n_1, n_2, ..., n_k} 
        such that n_1^x + n_2^x + ... + n_k^x = n.

        Args:
          n: The target integer sum.
          x: The power to which unique integers are raised.

        Returns:
          The number of ways modulo 10^9 + 7.
        """
        
        # Define the modulo constant
        MOD = 10**9 + 7

        # Initialize the DP table. dp[i] will store the number of ways 
        # to form the sum 'i' using unique x-th powers of integers 
        # considered so far.
        # The size is n + 1 to accommodate sums from 0 to n.
        dp = [0] * (n + 1)
        
        # Base case: There is exactly one way to form a sum of 0, 
        # which is by using no numbers.
        dp[0] = 1

        # Iterate through possible base numbers (num = 1, 2, 3, ...)
        # We only need to consider base numbers 'num' such that num^x <= n.
        num = 1
        while True:
            # Calculate the x-th power of the current base number 'num'.
            try:
                # Use pow() for potentially large intermediate results, 
                # although num**x is likely sufficient given constraints.
                p = pow(num, x) 
            except OverflowError:
                 # If calculating num^x overflows, it certainly exceeds n.
                 # This check might be redundant given the p > n check below,
                 # but adds robustness.
                break 

            # If the calculated power 'p' is greater than the target sum 'n',
            # then this power and any subsequent powers (from num+1, num+2, ...)
            # cannot be part of a sum that equals 'n'. We can stop the process
            # of considering further base numbers.
            if p > n:
                break

            # Iterate backwards through the DP table, from the target sum 'n'
            # down to the current power 'p'.
            # The backward iteration is crucial for the "unique integers" constraint.
            # It ensures that for the current power 'p' (derived from base 'num'),
            # we consider adding it only to sums that were formed *without* using 'p' yet
            # (i.e., using only bases less than 'num' in previous outer loop iterations).
            # If we iterated forwards, dp[j-p] might have already been updated using 'p' 
            # in this same outer loop iteration (for base 'num'), violating the uniqueness.
            for j in range(n, p - 1, -1):
                # Update the number of ways to form sum 'j'.
                # We add the number of ways to form the sum 'j - p' (using bases < num)
                # to the existing number of ways to form 'j' (also using bases < num).
                # This corresponds to including the current power 'p' in the sum.
                # Apply modulo at each addition to prevent overflow and keep results within range.
                dp[j] = (dp[j] + dp[j - p]) % MOD
            
            # Move to the next potential base number.
            num += 1

        # After iterating through all relevant base numbers, dp[n] will contain
        # the total number of ways to form the sum 'n' using unique x-th powers.
        return dp[n]