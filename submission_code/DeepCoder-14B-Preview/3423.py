class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        total = 0
        n = len(nums)
        for q in queries:
            pos, x = q
            nums[pos] = x
            if n == 0:
                current_max = 0
            else:
                include = nums[0]
                exclude = 0
                for i in range(1, n):
                    new_include = exclude + nums[i]
                    new_exclude = max(include, exclude)
                    include, exclude = new_include, new_exclude
                current_max = max(include, exclude)
            total += current_max
        return total % MOD