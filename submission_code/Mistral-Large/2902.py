from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def max_digit(n):
            return int(max(str(n)))

        max_sum = -1
        max_digits = [max_digit(num) for num in nums]

        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if max_digits[i] == max_digits[j]:
                    max_sum = max(max_sum, nums[i] + nums[j])

        return max_sum