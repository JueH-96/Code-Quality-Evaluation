from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Prefix sums  (prefix[i] = sum of nums[0..i-1])
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        NEG = -10**18                     # value smaller than any possible answer
        dp = [[NEG] * (k + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = 0                  # zero sub-arrays yield sum 0

        # dp[i][j] = best sum we can get from the first i elements
        #            using exactly j non-overlapping sub-arrays
        for j in range(1, k + 1):
            best = NEG                    # best value of (dp[l][j-1] - prefix[l])
            for i in range(1, n + 1):
                l = i - m                 # smallest l that can start a subarray ending at i
                if l >= 0:
                    cand = dp[l][j - 1] - prefix[l]
                    if cand > best:
                        best = cand

                # Option 1: do not end a subarray at position i
                dp[i][j] = dp[i - 1][j]

                # Option 2: end the j-th subarray at position i
                if best != NEG:
                    possible = prefix[i] + best
                    if possible > dp[i][j]:
                        dp[i][j] = possible

        return dp[n][k]