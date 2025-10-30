from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def max_sum_subsequence(arr):
            incl = 0
            excl = 0
            for i in arr:
                new_excl = max(excl, incl)
                incl = excl + i
                excl = new_excl
            return max(excl, incl)
        
        result = 0
        for pos, x in queries:
            original_val = nums[pos]
            nums[pos] = x
            result += max_sum_subsequence(nums)
            result %= MOD
            nums[pos] = original_val
        
        return result