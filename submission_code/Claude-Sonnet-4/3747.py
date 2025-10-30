class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        max_diff = 0
        n = len(nums)
        
        # Check all regular adjacent pairs
        for i in range(n - 1):
            max_diff = max(max_diff, abs(nums[i + 1] - nums[i]))
        
        # Check the circular adjacent pair (last and first elements)
        max_diff = max(max_diff, abs(nums[0] - nums[n - 1]))
        
        return max_diff