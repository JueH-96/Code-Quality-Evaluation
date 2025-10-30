from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0

        # Precompute smallest prime factor (spf) up to max(nums)
        maxv = max(nums)
        spf = list(range(maxv + 1))
        spf[0] = 0
        if maxv >= 1:
            spf[1] = 1
        p = 2
        while p * p <= maxv:
            if spf[p] == p:  # p is prime
                for multiple in range(p * p, maxv + 1, p):
                    if spf[multiple] == multiple:
                        spf[multiple] = p
            p += 1

        # For each x in nums, the value after operation is its smallest prime factor
        b = [spf[x] for x in nums]

        # dp[i][0] = min ops to make nums[0..i] non-decreasing and not operate on i
        # dp[i][1] = same, but operate on i (i.e., use b[i])
        INF = 10**18
        dp0, dp1 = 0, 1  # for i = 0
        # No previous constraint for i = 0; using b[0] costs 1 op

        for i in range(1, n):
            a_prev0 = nums[i-1]
            a_prev1 = b[i-1]
            a_curr0 = nums[i]
            a_curr1 = b[i]

            ndp0 = INF
            ndp1 = INF

            # Transition from state 0 at i-1 (used nums[i-1])
            if dp0 < INF:
                if a_prev0 <= a_curr0:
                    ndp0 = min(ndp0, dp0)
                if a_prev0 <= a_curr1:
                    ndp1 = min(ndp1, dp0 + 1)

            # Transition from state 1 at i-1 (used b[i-1])
            if dp1 < INF:
                if a_prev1 <= a_curr0:
                    ndp0 = min(ndp0, dp1)
                if a_prev1 <= a_curr1:
                    ndp1 = min(ndp1, dp1 + 1)

            dp0, dp1 = ndp0, ndp1

            # If both are INF, early exit
            if dp0 == INF and dp1 == INF:
                return -1

        res = min(dp0, dp1)
        return -1 if res >= INF else res