class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        max_val = nums[-1]
        current_sum = nums[-1]
        for i in range(n-2, -1, -1):
            if nums[i] <= current_sum:
                current_sum += nums[i]
            else:
                current_sum = nums[i]
            if current_sum > max_val:
                max_val = current_sum
        return max_val