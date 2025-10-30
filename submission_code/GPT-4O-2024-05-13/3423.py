from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def max_non_adjacent_sum(arr):
            incl = 0
            excl = 0
            for num in arr:
                new_excl = max(incl, excl)
                incl = excl + num
                excl = new_excl
            return max(incl, excl)
        
        total_sum = 0
        
        for pos, x in queries:
            nums[pos] = x
            total_sum += max_non_adjacent_sum(nums)
            total_sum %= MOD
        
        return total_sum