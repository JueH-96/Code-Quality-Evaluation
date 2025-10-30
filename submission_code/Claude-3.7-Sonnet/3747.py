class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        n = len(nums)
        max_diff = 0
        
        for i in range(n):
            next_idx = (i + 1) % n  # Handles the circular property
            diff = abs(nums[i] - nums[next_idx])
            max_diff = max(max_diff, diff)
            
        return max_diff