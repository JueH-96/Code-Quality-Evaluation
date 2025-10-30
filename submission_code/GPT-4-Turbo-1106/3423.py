class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def max_sum_no_adjacent(arr):
            incl = 0
            excl = 0
            for num in arr:
                new_excl = max(excl, incl)
                incl = excl + num
                excl = new_excl
            return max(incl, excl)
        
        sum_answers = 0
        for pos, x in queries:
            nums[pos] = x
            sum_answers += max_sum_no_adjacent(nums)
            sum_answers %= MOD
        
        return sum_answers