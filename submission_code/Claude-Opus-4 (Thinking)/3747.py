class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        max_diff = 0
        n = len(nums)
        
        # Check all adjacent pairs including the circular connection
        for i in range(n):
            # Next index in circular array (wraps around)
            next_i = (i + 1) % n
            diff = abs(nums[i] - nums[next_i])
            max_diff = max(max_diff, diff)
        
        return max_diff