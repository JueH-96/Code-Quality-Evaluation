class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7

        # dp[j] will store the number of ways to form a sum of j.
        dp = [0] * (n + 1)
        
        # Base case: There's one way to form a sum of 0, which is by choosing no numbers.
        dp[0] = 1
        
        # We can use numbers num = 1, 2, 3, ... such that num^x <= n.
        # Iterate through these base numbers.
        num = 1
        while True:
            # Calculate the current number's power.
            power = pow(num, x)
            
            # If the power is greater than n, we cannot use this number or any larger ones,
            # as their powers would also be greater than n. So, we can stop.
            if power > n:
                break
            
            # This problem is equivalent to the 0/1 knapsack problem. We update the dp table
            # by considering the current item (power). To ensure that each item is used at
            # most once (due to the "unique integers" constraint), we iterate the sum j
            # from n down to the item's value (power).
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % MOD
            
            # Move to the next integer.
            num += 1
                
        # The final answer is the number of ways to form the sum n.
        return dp[n]