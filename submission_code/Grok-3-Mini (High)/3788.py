class Solution:
    def maxSum(self, nums: List[int]) -> int:
        unique_vals = set(nums)
        pos_sum = sum(val for val in unique_vals if val > 0)
        if pos_sum > 0:
            return pos_sum
        else:
            return max(nums)