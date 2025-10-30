class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        min_diff = float('inf')
        for i in range(len(nums) - 1):
            diff = nums[i + 1] - nums[i]
            min_diff = min(min_diff, diff)
        return min_diff