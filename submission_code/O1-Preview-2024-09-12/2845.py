class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        min_value = float('inf')
        for i in range(1, len(nums)):
            min_value = min(min_value, nums[i] - nums[i-1])
        return min_value