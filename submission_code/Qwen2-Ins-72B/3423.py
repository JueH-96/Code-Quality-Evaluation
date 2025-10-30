from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        queries = sorted([(pos, x, i) for i, (pos, x) in enumerate(queries)], key=lambda x: x[0])
        dp = [0] * (n + 1)
        ans = [0] * len(queries)
        idx = 0
        for pos, x, i in queries:
            while idx < pos:
                dp[idx + 1] = max(dp[idx], dp[idx + 1]) + nums[idx]
                idx += 1
            dp[idx + 1] = max(dp[idx], dp[idx + 1]) + x
            ans[i] = max(dp[:idx + 2])
            nums[pos] = x
        while idx < n:
            dp[idx + 1] = max(dp[idx], dp[idx + 1]) + nums[idx]
            idx += 1
        return sum(ans) % MOD