from typing import List

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        x_sub = x
        cost = []
        for s in range(n - x_sub + 1):
            sub = nums[s:s+x_sub]
            sorted_sub = sorted(sub)
            m = (x_sub - 1) // 2
            median = sorted_sub[m]
            # compute prefix sum
            prefix = [0]
            for num in sorted_sub:
                prefix.append(prefix[-1] + num)
            sum_abs = (median * (m + 1) - prefix[m + 1] + 
                       (prefix[x_sub] - prefix[m + 1]) - 
                       median * (x_sub - m - 1))
            cost.append(sum_abs)
        
        INF = float('inf')
        # dp[i][j]: min cost for first i elements, j subarrays
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                # option 1: not take subarray ending at i-1
                option1 = dp[i-1][j]
                option2 = INF
                if i >= x_sub:
                    s = i - x_sub
                    if 0 <= s < len(cost):
                        option2 = dp[s][j-1] + cost[s]
                dp[i][j] = min(option1, option2)
        
        return dp[n][k]