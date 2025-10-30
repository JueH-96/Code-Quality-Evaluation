from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        """
        Dynamic programming.
        dp[s] – the maximum number of elements of a subsequence (among the
        elements processed so far) whose sum equals s.
        We scan the array from left to right; for every number we try to add it
        to all previously achievable sums, iterating backwards so each element
        is used at most once (respecting the subsequence/order constraint).
        """
        NEG = -10**9                       # sentinel for “unreachable”
        dp = [NEG] * (target + 1)
        dp[0] = 0                          # empty subsequence has sum 0, length 0

        for num in nums:
            if num > target:               # too large to ever be part of a sum ≤ target
                continue
            for s in range(target, num - 1, -1):   # iterate backwards
                if dp[s - num] != NEG:             # sum (s-num) is achievable
                    dp[s] = max(dp[s], dp[s - num] + 1)

        return dp[target] if dp[target] != NEG else -1