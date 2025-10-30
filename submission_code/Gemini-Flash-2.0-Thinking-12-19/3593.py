import math

class Solution:
    def gcd_two(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def gcd_list(self, numbers):
        if not numbers:
            return 0
        result = numbers[0]
        for i in range(1, len(numbers)):
            result = self.gcd_two(result, numbers[i])
        return result

    def lcm_two(self, a, b):
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // self.gcd_two(a, b)

    def lcm_list(self, numbers):
        if not numbers:
            return 0
        result = numbers[0]
        for i in range(1, len(numbers)):
            result = self.lcm_two(result, numbers[i])
        return result

    def factor_score(self, numbers):
        if not numbers:
            return 0
        if len(numbers) == 1:
            return numbers[0] * numbers[0]
        gcd_val = self.gcd_list(numbers)
        lcm_val = self.lcm_list(numbers)
        return gcd_val * lcm_val

    def maxScore(self, nums: List[int]) -> int:
        max_score_val = 0
        max_score_val = max(max_score_val, self.factor_score(nums))
        for i in range(len(nums)):
            temp_nums = nums[:i] + nums[i+1:]
            max_score_val = max(max_score_val, self.factor_score(temp_nums))
        return max_score_val