from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # ---------------------------------------------------------------------
        # 1) We will precompute the smallest prime factor (spf) for every
        #    number from 1 to 10^6 using a sieve-like method.
        #    Then spf[x] = x if x is prime, and spf[x] < x if x is composite.
        #
        # 2) For each nums[i], define:
        #       hi[i] = nums[i]
        #       li[i] = spf(nums[i]) if spf(nums[i]) < nums[i], else nums[i]
        #
        #    The cost to use li[i] (i.e. perform one operation) is:
        #       costLi[i] = 1 if li[i] < hi[i], else 0
        #
        # 3) We use dynamic programming to decide for each position i whether
        #    we keep hi[i] (0 operations) or use li[i] (possibly 1 operation),
        #    subject to maintaining a non-decreasing final array. Let:
        #
        #       dp[i][0] = min total cost if we end with final[i] = hi[i]
        #       dp[i][1] = min total cost if we end with final[i] = li[i]
        #
        #    Recurrence:
        #
        #       dp[i][0] = min of dp[i-1][0] if hi[i-1] <= hi[i],
        #                              dp[i-1][1] if li[i-1] <= hi[i]
        #
        #       dp[i][1] = min of dp[i-1][0] if hi[i-1] <= li[i],
        #                              dp[i-1][1] if li[i-1] <= li[i]
        #                   plus costLi[i]
        #
        # 4) The answer is min(dp[n-1][0], dp[n-1][1]) or -1 if neither is valid.
        #
        # 5) Time complexity: O(n + max(nums)) which is feasible for n=10^5,
        #    max(nums)=10^6.
        # ---------------------------------------------------------------------

        MAX_VAL = 10**6

        # Build array spf[] where spf[x] = smallest prime factor of x
        # For a prime p, spf[p] = p.
        spf = [0] * (MAX_VAL + 1)
        spf[1] = 1  # define spf(1) = 1 for convenience
        for i in range(2, MAX_VAL + 1):
            if spf[i] == 0:  # i is prime
                spf[i] = i
                for j in range(i * 2, MAX_VAL + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

        n = len(nums)

        # Precompute hi[i], li[i], and costLi[i]
        hi = [0] * n
        li = [0] * n
        costLi = [0] * n
        for i in range(n):
            x = nums[i]
            hi[i] = x
            if x == 1:
                # spf(1) = 1 => only one possible value
                li[i] = 1
                costLi[i] = 0
            else:
                # If spf[x] < x, x is composite => dividing by greatest proper divisor
                # results in spf(x). Otherwise, it's prime => no reduction possible.
                if spf[x] < x:
                    li[i] = spf[x]
                    costLi[i] = 1
                else:
                    li[i] = x
                    costLi[i] = 0

        # DP arrays for cost: dp0 = dp[i][0], dp1 = dp[i][1]
        INF = 10**15
        dp0, dp1 = 0, costLi[0]  # base case for i=0

        for i in range(1, n):
            newdp0, newdp1 = INF, INF

            # final[i] = hi[i]
            # we can come from dp0 if hi[i-1] <= hi[i]
            # or from dp1 if li[i-1] <= hi[i]
            if hi[i - 1] <= hi[i]:
                newdp0 = min(newdp0, dp0)
            if li[i - 1] <= hi[i]:
                newdp0 = min(newdp0, dp1)

            # final[i] = li[i], costLi[i] is added if li[i] < hi[i]
            # we can come from dp0 if hi[i-1] <= li[i]
            # or from dp1 if li[i-1] <= li[i]
            if hi[i - 1] <= li[i]:
                newdp1 = min(newdp1, dp0 + costLi[i])
            if li[i - 1] <= li[i]:
                newdp1 = min(newdp1, dp1 + costLi[i])

            dp0, dp1 = newdp0, newdp1

            if dp0 == INF and dp1 == INF:
                # Not possible to continue in a non-decreasing manner
                return -1

        ans = min(dp0, dp1)
        return ans if ans < INF else -1