from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        
        # dp[r][s] = number of ways to pick exactly r elements from nums
        # (processed so far) that sum to s
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1  # One way to choose 0 elements with sum 0
        
        # Build the dp table
        for num in nums:
            # Update in descending order to avoid counting an element multiple times
            for r in range(n, 0, -1):
                for s in range(k, num - 1, -1):
                    dp[r][s] = (dp[r][s] + dp[r - 1][s - num]) % mod
        
        # Precompute powers of 2 for summation
        power_of_two = [1] * (n + 1)
        for i in range(1, n + 1):
            power_of_two[i] = (power_of_two[i - 1] * 2) % mod
        
        # Sum over all r of (dp[r][k] * 2^(n-r))
        result = 0
        for r in range(n + 1):
            result = (result + dp[r][k] * power_of_two[n - r]) % mod
        
        return result