from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i][d] = longest subseq ending at i with last abs difference exactly d
        # best[i][d] = max(dp[i][k] for k >= d), for d in [0..300]
        MAXD = 300
        dp = [ [0] * (MAXD+1) for _ in range(n) ]
        best = [ [0] * (MAXD+2) for _ in range(n) ]  # extra slot for convenience
        
        ans = 1
        for i in range(n):
            # compare i with all previous j
            ai = nums[i]
            dpi = dp[i]
            for j in range(i):
                diff = abs(ai - nums[j])
                # we can extend any subseq ending at j whose last diff >= diff
                val = 1 + best[j][diff]
                if val > dpi[diff]:
                    dpi[diff] = val
                    if val + 1 > ans:
                        # +1 because dp counts number of diffs, subseq length = diffs+1
                        ans = val + 1
            # build best[i] as suffix max over dp[i]
            bi = best[i]
            # best[i][MAXD+1] = 0 already
            cur = 0
            for d in range(MAXD, -1, -1):
                if dp[i][d] > cur:
                    cur = dp[i][d]
                bi[d] = cur
        
        return ans