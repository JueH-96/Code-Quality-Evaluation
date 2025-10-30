class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        def solve(arr):
            include = 0
            exclude = 0
            for num in arr:
                new_include = exclude + max(0, num)
                new_exclude = max(include, exclude)
                include = new_include
                exclude = new_exclude
            return max(include, exclude)

        total_sum = 0
        for pos, x in queries:
            original_val = nums[pos]
            nums[pos] = x
            total_sum = (total_sum + solve(nums)) % (10**9 + 7)
            
        return total_sum