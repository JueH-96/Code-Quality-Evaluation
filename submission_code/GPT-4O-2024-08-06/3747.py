class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        max_diff = 0
        n = len(nums)
        
        for i in range(n):
            # Calculate the absolute difference between current and next element
            diff = abs(nums[i] - nums[(i + 1) % n])
            # Update max_diff if the current difference is greater
            max_diff = max(max_diff, diff)
        
        return max_diff