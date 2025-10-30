class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        current = nums[-1]
        max_val = current
        for i in range(n - 2, -1, -1):
            if nums[i] <= current:
                current += nums[i]
            else:
                current = nums[i]
            if current > max_val:
                max_val = current
        return max_val