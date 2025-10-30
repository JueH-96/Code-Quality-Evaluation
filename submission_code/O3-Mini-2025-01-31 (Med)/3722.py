from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        # prefix sum: p[i] = sum(nums[0..i-1])
        p = [0] * (n + 1)
        for i in range(1, n + 1):
            p[i] = p[i-1] + nums[i-1]
        
        # dp[j][i] will be the maximum sum using exactly j subarrays 
        # from the first i numbers (i elements, indices 0..i-1)
        # For j = 0, use 0 for any i.
        # For j > 0, it is impossible to form subarrays if i < j*m.
        NEG_INF = float('-inf')
        dp = [[NEG_INF] * (n + 1) for _ in range(k + 1)]
        for i in range(n + 1):
            dp[0][i] = 0  # zero segments gives sum=0

        # for each count of subarrays from 1 to k.
        for j in range(1, k + 1):
            # For dp[j][i], the minimum i is j * m
            # Use best to maintain maximum value of dp[j-1][t] - p[t] for valid t.
            best = NEG_INF
            for i in range(j * m, n + 1):
                # Before considering i, update best using index i-m:
                # The idea: if we choose a new subarray ending at i-1,
                # then it must start at some t where t <= i-m.
                # So we update best = max(best, dp[j-1][t] - p[t]) for t = i-m.
                t = i - m
                # Only update best if dp[j-1][t] is defined
                if dp[j-1][t] != NEG_INF:
                    best = max(best, dp[j-1][t] - p[t])
                # candidate if we end a new segment exactly at i-1,
                # which has sum = p[i] - p[t], with best giving optimal dp[j-1][t] - p[t]
                candidate = best + p[i] if best != NEG_INF else NEG_INF
                # Also, it's allowed to not use i-1 (carry over the best state from i-1)
                dp[j][i] = max(dp[j][i-1], candidate)
        
        return dp[k][n]