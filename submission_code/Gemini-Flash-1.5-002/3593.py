import math

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)

        def lcm(a, b):
            return (a * b) // gcd(a, b)

        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0] ** 2

        max_score = 0
        for i in range(n + 1):
            temp_nums = []
            if i < n:
                temp_nums = nums[:i] + nums[i+1:]
            
            if not temp_nums:
                max_score = max(max_score,0)
                continue

            current_gcd = temp_nums[0]
            current_lcm = temp_nums[0]
            for j in range(1, len(temp_nums)):
                current_gcd = gcd(current_gcd, temp_nums[j])
                current_lcm = lcm(current_lcm, temp_nums[j])
            max_score = max(max_score, current_gcd * current_lcm)

        return max_score