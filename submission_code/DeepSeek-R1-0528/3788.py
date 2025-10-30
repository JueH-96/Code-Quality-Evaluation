class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_val = max(nums)
        if max_val <= 0:
            return max_val
        return sum(set(x for x in nums if x > 0))