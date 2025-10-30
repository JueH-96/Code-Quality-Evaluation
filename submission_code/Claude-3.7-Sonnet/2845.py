class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array
        nums.sort()
        
        # Find the minimum difference between adjacent elements
        min_diff = float('inf')
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i-1]
            min_diff = min(min_diff, diff)
        
        return min_diff