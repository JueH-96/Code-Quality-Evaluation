class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        max_diff = 0
        n = len(nums)
        
        for i in range(n):
            # Calculate the absolute difference between adjacent elements
            # Considering circular nature, nums[n-1] is adjacent to nums[0]
            diff = abs(nums[i] - nums[(i + 1) % n])
            max_diff = max(max_diff, diff)
        
        return max_diff