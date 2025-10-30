from typing import List
import math

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        INF = 10**18
        # dp[k][i]: minimal sum for first k segments, next segment starts at i
        dp = [ [INF] * (n+1) for _ in range(m+1) ]
        dp[0][0] = 0
        
        for k in range(m):
            target = andValues[k]
            for start in range(n):
                if dp[k][start] == INF:
                    continue
                # try to find subarray [start..r] with AND == target
                cur = (1<<17)-1  # all bits 1 up to max nums bit
                # for target==0 we will note first zero pos
                first_zero = -1
                for r in range(start, n):
                    cur &= nums[r]
                    # if we lost any bit in target, we can never match
                    if (cur & target) != target:
                        break
                    if cur == target:
                        dp[k+1][r+1] = min(dp[k+1][r+1], dp[k][start] + nums[r])
                        # record the first time it becomes zero, to allow further zero matches
                        if target == 0 and first_zero < 0:
                            first_zero = r
                # If target is 0 and we saw AND==0 at r0, then for all r>r0, cur remains 0
                # we need to update those
                if target == 0 and first_zero >= 0:
                    base = dp[k][start]
                    # we've already done r==first_zero, do the rest
                    for r in range(first_zero+1, n):
                        dp[k+1][r+1] = min(dp[k+1][r+1], base + nums[r])
            # after filling dp[k+1], move on
        # answer is min over dp[m][i]
        ans = min(dp[m])
        return ans if ans < INF else -1