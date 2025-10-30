from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def lcm(a, b):
            return (a * b) // gcd(a, b)

        def gcd_of_list(numbers):
            if not numbers:
                return 1
            result = numbers[0]
            for i in range(1, len(numbers)):
                result = gcd(result, numbers[i])
            return result

        def lcm_of_list(numbers):
            if not numbers:
                return 1
            result = numbers[0]
            for i in range(1, len(numbers)):
                result = lcm(result, numbers[i])
            return result

        def factor_score(numbers):
            if not numbers:
                return 0
            g = gcd_of_list(numbers)
            l = lcm_of_list(numbers)
            return g * l

        max_score = 0

        # Case 1: No element removed
        max_score = max(max_score, factor_score(nums))

        # Case 2: Remove one element at a time
        for i in range(len(nums)):
            temp_nums = nums[:i] + nums[i+1:]
            max_score = max(max_score, factor_score(temp_nums))

        return max_score