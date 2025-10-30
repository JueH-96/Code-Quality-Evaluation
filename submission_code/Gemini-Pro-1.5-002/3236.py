class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        n = len(nums)
        longest_prefix_sum = 0
        current_prefix_sum = 0
        current_prefix_length = 0
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1] + 1:
                current_prefix_sum += nums[i]
                current_prefix_length += 1
            elif i == 0:
                current_prefix_sum += nums[i]
                current_prefix_length += 1
            else:
                longest_prefix_sum = max(longest_prefix_sum, current_prefix_sum)
                current_prefix_sum = nums[i]
                current_prefix_length = 1
        longest_prefix_sum = max(longest_prefix_sum, current_prefix_sum)

        x = longest_prefix_sum
        while x in nums:
            x += 1
        return x