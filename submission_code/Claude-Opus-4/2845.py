class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array
        nums.sort()
        
        # Find minimum difference between consecutive elements
        min_diff = float('inf')
        
        for i in range(len(nums) - 1):
            diff = nums[i + 1] - nums[i]
            min_diff = min(min_diff, diff)
        
        return min_diff