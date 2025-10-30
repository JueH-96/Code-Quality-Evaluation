from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        # prefix sums
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        # dp for j=0: dp[i]=0 for all i (zero segments => sum 0)
        prev_dp = [0] * (n + 1)
        NEG_INF = -10**18
        
        # build dp for j=1..k
        for _ in range(1, k + 1):
            curr_dp = [NEG_INF] * (n + 1)
            best = NEG_INF
            # curr_dp[0] should stay NEG_INF (can't have >0 segments in 0 items)
            for i in range(1, n + 1):
                # option: do not end a segment at i
                curr_dp[i] = curr_dp[i - 1]
                # option: end a segment at i of length >= m
                if i >= m:
                    t = i - m
                    # update best = max(prev_dp[t] - prefix[t]) for t up to i-m
                    val = prev_dp[t] - prefix[t]
                    if val > best:
                        best = val
                    # if best is valid, form a segment ending at i
                    candidate = prefix[i] + best
                    if candidate > curr_dp[i]:
                        curr_dp[i] = candidate
            # move to next layer
            prev_dp = curr_dp
        
        # answer is dp[n][k]
        return prev_dp[n]