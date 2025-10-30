class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        return min(sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1))