class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7

        # dp[i] will store the number of ways to achieve sum i
        # using x-th powers of unique positive integers processed so far.
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: sum 0 can be achieved in one way (by choosing an empty set of numbers)

        num = 1
        while True:
            # Calculate value of num^x
            # Using built-in pow(base, exp) is efficient for integers.
            term_val = pow(num, x)

            if term_val > n:
                # If num^x exceeds n, this term and any subsequent terms 
                # (from (num+1)^x, etc.) are too large to be part of the sum.
                # So, we can stop processing further numbers.
                break

            # Update dp table: iterate downwards to ensure each num^x is used at most once.
            # This is a standard technique for 0/1 knapsack-type problems.
            # For each target sum j from n down to term_val (inclusive):
            #   dp[j] can be formed in two ways:
            #     1. By not including term_val: The number of ways is dp[j] (calculated using previous numbers).
            #     2. By including term_val: The number of ways is dp[j - term_val] (ways to make sum j-term_val using previous numbers).
            #   So, the new dp[j] is the sum of these two possibilities.
            for j in range(n, term_val - 1, -1):
                dp[j] = (dp[j] + dp[j - term_val]) % MOD
            
            num += 1
            # The loop is guaranteed to terminate because num increases, 
            # and thus term_val = num^x also increases (for x >= 1).
            # Eventually, term_val will exceed n.
        
        return dp[n]