class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        mod = 10**9 + 7
        total_sum = 0
        for query in queries:
            pos, x = query
            nums[pos] = x
            incl = 0
            excl = 0
            for num in nums:
                new_excl = max(incl, excl)
                incl = excl + num
                excl = new_excl
            total_sum = (total_sum + max(incl, excl)) % mod
        return total_sum