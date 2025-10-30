from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        INF = float('-inf')
        # dp[i][j] represents the maximum sum for the first i elements with j subarrays
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = 0  # Base case: 0 subarrays have sum 0
        
        for j in range(1, k + 1):
            max_so_far = INF
            for i in range(1, n + 1):
                required = j * m
                if i < required:
                    dp[i][j] = INF
                else:
                    if i >= m:
                        current_s = i - m
                        # Calculate candidate value for the current_s
                        candidate = dp[current_s][j-1] - prefix[current_s]
                        if candidate > max_so_far:
                            max_so_far = candidate
                    # Now compute option2
                    if max_so_far == INF:
                        option2 = INF
                    else:
                        option2 = max_so_far + prefix[i]
                    # option1 is the value without taking a new subarray ending at i-1
                    option1 = dp[i-1][j]
                    dp[i][j] = max(option1, option2)
        return dp[n][k]