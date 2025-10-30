class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array to easily find the minimum difference
        nums.sort()
        
        # Initialize the minimum difference to a large value
        min_diff = float('inf')
        
        # Iterate through the sorted array and find the minimum difference
        for i in range(1, len(nums)):
            # Calculate the difference between consecutive elements
            diff = nums[i] - nums[i - 1]
            # Update the minimum difference if the current difference is smaller
            min_diff = min(min_diff, diff)
        
        return min_diff