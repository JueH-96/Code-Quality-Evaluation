class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array
        nums.sort()
        # Initialize the minimum difference to a large number
        min_diff = float('inf')
        # Iterate through the sorted array to find the minimum difference between consecutive elements
        for i in range(len(nums) - 1):
            diff = nums[i + 1] - nums[i]
            if diff < min_diff:
                min_diff = diff
        return min_diff