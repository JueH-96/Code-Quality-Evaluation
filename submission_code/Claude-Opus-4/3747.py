class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        n = len(nums)
        max_diff = 0
        
        # Check all consecutive pairs
        for i in range(n - 1):
            diff = abs(nums[i] - nums[i + 1])
            max_diff = max(max_diff, diff)
        
        # Check the circular connection (last and first element)
        diff = abs(nums[n - 1] - nums[0])
        max_diff = max(max_diff, diff)
        
        return max_diff