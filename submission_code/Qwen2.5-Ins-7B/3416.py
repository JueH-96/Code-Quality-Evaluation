from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        total_diff = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                num1, num2 = str(nums[i]), str(nums[j])
                diff = sum(a != b for a, b in zip(num1, num2))
                total_diff += diff
        return total_diff