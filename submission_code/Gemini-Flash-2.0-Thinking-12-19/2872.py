class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        current_sum = nums[-1]
        max_sum = current_sum
        for i in range(n - 2, -1, -1):
            if nums[i] <= current_sum:
                current_sum += nums[i]
            else:
                current_sum = nums[i]
            max_sum = max(max_sum, current_sum)
        return max_sum