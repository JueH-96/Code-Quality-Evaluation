class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        max_num = max(nums)
        max_num_count = nums.count(max_num)
        return max_num_count