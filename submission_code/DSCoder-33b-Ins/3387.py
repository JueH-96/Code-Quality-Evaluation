class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        median = nums[len(nums) // 2]
        return sum(abs(k - num) for num in nums)