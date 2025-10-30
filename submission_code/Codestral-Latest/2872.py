class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        result = nums[-1]
        current_sum = nums[-1]

        for i in range(n - 2, -1, -1):
            if nums[i] <= current_sum:
                current_sum += nums[i]
            else:
                current_sum = nums[i]
            result = max(result, current_sum)

        return result