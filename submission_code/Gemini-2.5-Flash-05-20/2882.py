import math

class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        # Define the modulo constant
        MOD = 10**9 + 7

        # dp[s] will store the number of ways to express sum s
        # using x-th powers of unique positive integers.
        # Initialize dp array with zeros.
        dp = [0] * (n + 1)
        
        # There is one way to make sum 0: by choosing no numbers.
        dp[0] = 1  

        # Iterate through possible positive integers 'i' whose x-th power
        # can potentially be part of the sum.
        # We start from i = 1.
        i = 1
        while True:
            # Calculate the x-th power of the current integer 'i'.
            current_power = i ** x
            
            # If the current x-th power exceeds 'n', it means we cannot use
            # this power or any larger powers (e.g., (i+1)^x) to form 'n'
            # without exceeding the target sum. So, we break the loop.
            if current_power > n:
                break
            
            # Iterate 's' from 'n' down to 'current_power'.
            # Iterating downwards is crucial for ensuring that each x-th power
            # (current_power) is used at most once in any given sum.
            # This is analogous to the 0/1 knapsack problem.
            for s in range(n, current_power - 1, -1):
                # The number of ways to form sum 's' is updated by adding:
                # 1. The ways to form 's' without using 'current_power' (which is dp[s] before this update).
                # 2. The ways to form 's - current_power' using integers whose x-th powers
                #    were considered *before* 'current_power'. By adding 'current_power' to these sums,
                #    we get new ways to form 's'.
                dp[s] = (dp[s] + dp[s - current_power]) % MOD
            
            # Move to the next integer.
            i += 1
            
        # The final result is the number of ways to form sum 'n'.
        return dp[n]