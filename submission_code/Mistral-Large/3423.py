from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7

        def maxSubsequenceSum(arr: List[int]) -> int:
            prev_prev, prev = 0, 0
            for num in arr:
                prev_prev, prev = prev, max(prev, prev_prev + max(0, num))
            return prev

        total_sum = 0
        for pos, x in queries:
            nums[pos] = x
            total_sum += maxSubsequenceSum(nums)
            total_sum %= MOD

        return total_sum