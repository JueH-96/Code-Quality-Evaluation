class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_diff = float('inf')
        
        for i in range(n - x):
            for j in range(i + x, n):
                min_diff = min(min_diff, abs(nums[i] - nums[j]))
        
        return min_diff