from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def max_sum_non_adjacent(arr):
            incl = 0
            excl = 0
            for num in arr:
                new_excl = max(incl, excl)
                incl = excl + num
                excl = new_excl
            return max(incl, excl)
        
        result = 0
        for pos, x in queries:
            nums[pos] = x
            result += max_sum_non_adjacent(nums)
            result %= MOD
        
        return result