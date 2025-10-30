from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        # dp[i][v] counts how many ways to choose arr1[0..i] ending with value v.
        # Value v can be from 0 to 50.
        dp = [[0] * 51 for _ in range(n)]
        
        # Initialize for i == 0: arr1[0] can be any value from 0 to nums[0] (since arr2[0] = nums[0]-arr1[0] is automatically valid)
        for v in range(0, nums[0] + 1):
            dp[0][v] = 1
        
        # Fill dp for subsequent positions
        for i in range(0, n - 1):
            # Let gap_i be the minimum required difference between arr1[i+1] and arr1[i]
            # We require: arr1[i+1] - arr1[i] >= delta, where delta = max(0, nums[i+1] - nums[i])
            delta = max(0, nums[i+1] - nums[i])
            
            # Instead of iterating for each possible current value and then for each next value,
            # We can precompute prefix sums for dp[i] for efficient summation.
            prefix = [0] * 52  # one extra for ease of calculation.
            # Build cumulative sum: prefix[v+1] = sum_{u=0}^{v} dp[i][u]
            for v in range(51):
                prefix[v+1] = (prefix[v] + dp[i][v]) % mod
                
            for current in range(51):
                count_current = dp[i][current]
                if count_current:
                    # The allowed next value must be at least: current + delta.
                    low = current + delta
                    # Also, next value cannot exceed nums[i+1]
                    high = nums[i+1]
                    if low > high:
                        continue
                    # Instead of iterating through next values one by one, add directly:
                    # But note, we must add dp[i][current] to each dp[i+1][next] for next in [low, high].
                    # To be efficient, we can later do another cumulative summing method.
                    # Here, since constraints are small (v up to 50), a direct loop is acceptable.
                    for nxt in range(low, high + 1):
                        dp[i+1][nxt] = (dp[i+1][nxt] + count_current) % mod
        
        # Sum over all valid endings at index n-1
        result = 0
        for v in range(51):
            result = (result + dp[n-1][v]) % mod
        return result