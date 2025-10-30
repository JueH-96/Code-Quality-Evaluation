class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median = nums[n//2] if n % 2 == 1 else max(nums[n//2 - 1], nums[n//2])
        return abs(median - k)