from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = -1
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                num1 = nums[i]
                num2 = nums[j]
                max_digit1 = max(int(d) for d in str(num1))
                max_digit2 = max(int(d) for d in str(num2))
                if max_digit1 == max_digit2:
                    current_sum = num1 + num2
                    if current_sum > max_sum:
                        max_sum = current_sum
        return max_sum if max_sum != -1 else -1