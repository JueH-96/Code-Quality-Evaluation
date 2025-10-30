class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median = nums[n // 2] if n % 2 == 1 else (nums[n // 2 - 1] + nums[n // 2]) / 2
        if median == k:
            return 0
        elif median < k:
            return sum(min(k - num, num - k) for num in nums[n // 2:])
        else:
            return sum(min(k - num, num - k) for num in nums[:n // 2])