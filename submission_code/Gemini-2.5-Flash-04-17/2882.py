class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # dp[j] will store the number of ways to form sum j
        # using the x-th powers of unique positive integers considered so far.
        # The size is n + 1 to include sums from 0 to n.
        dp = [0] * (n + 1)
        
        # Base case: There is one way to form sum 0 (by choosing no numbers).
        dp[0] = 1
        
        # Iterate through possible positive integers i starting from 1.
        # We consider adding i^x to the sum.
        i = 1
        while True:
            # Calculate the x-th power of the current integer i.
            # Use pow(base, exp) which is generally efficient and handles potential
            # large results (though here i^x <= n <= 300).
            current_power = pow(i, x)
            
            # If the current power exceeds n, we cannot use i^x or any larger power
            # (since (i+1)^x > i^x for positive x), so we can stop considering i and beyond.
            if current_power > n:
                break
                
            # Update the dp array. We iterate j from n down to current_power.
            # This order is crucial for the 0/1 knapsack-like logic to ensure that
            # each power i^x is used at most once in any combination for a given sum.
            # When we update dp[j] using dp[j - current_power], the value dp[j - current_power]
            # reflects the number of ways to achieve the sum (j - current_power) using
            # only powers of integers strictly less than i (because those were the values
            # present in the dp array before processing the current integer i).
            for j in range(n, current_power - 1, -1):
                # The number of ways to get sum j using powers up to i equals:
                # (ways to get sum j using powers up to i-1)  # This is the current dp[j]
                # + (ways to get sum j - current_power using powers up to i-1, then add current_power) # This is dp[j - current_power] from the previous iteration i-1
                dp[j] = (dp[j] + dp[j - current_power]) % MOD
            
            # Move to the next potential integer whose power we might use.
            i += 1
        
        # The final answer is the number of ways to form sum n using the unique powers considered.
        return dp[n]