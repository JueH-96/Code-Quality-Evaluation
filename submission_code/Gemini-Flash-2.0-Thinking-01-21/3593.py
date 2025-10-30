import math

class Solution:
    def gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(self, a, b):
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // self.gcd(a, b)

    def gcd_array(self, arr):
        if not arr:
            return 0
        if len(arr) == 1:
            return arr[0]
        res = arr[0]
        for i in range(1, len(arr)):
            res = self.gcd(res, arr[i])
        return res

    def lcm_array(self, arr):
        if not arr:
            return 0
        if len(arr) == 1:
            return arr[0]
        res = arr[0]
        for i in range(1, len(arr)):
            res = self.lcm(res, arr[i])
        return res

    def factor_score(self, arr):
        if not arr:
            return 0
        g = self.gcd_array(arr)
        l = self.lcm_array(arr)
        return g * l

    def maxScore(self, nums: List[int]) -> int:
        max_score = 0
        max_score = max(max_score, self.factor_score(nums))
        for i in range(len(nums)):
            temp_nums = nums[:i] + nums[i+1:]
            max_score = max(max_score, self.factor_score(temp_nums))
        return max_score