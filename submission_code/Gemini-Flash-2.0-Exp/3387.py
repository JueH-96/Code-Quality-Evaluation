class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = (n - 1) // 2
        
        return abs(nums[median_index] - k)