class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        n = len(nums)
        max_diff = 0
        
        for i in range(n):
            diff1 = abs(nums[i] - nums[(i+1) % n])
            diff2 = abs(nums[i] - nums[(i-1) % n])
            max_diff = max(max_diff, diff1, diff2)
        
        return max_diff