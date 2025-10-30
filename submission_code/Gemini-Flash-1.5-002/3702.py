import math

class Solution:
    def maxLength(self, nums: list[int]) -> int:
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)

        def lcm(a, b):
            return (a * b) // gcd(a, b)

        max_len = 0
        for i in range(len(nums)):
            prod = 1
            g = nums[i]
            l = nums[i]
            for j in range(i, len(nums)):
                prod *= nums[j]
                g = gcd(g, nums[j])
                l = lcm(l, nums[j])
                if prod == g * l:
                    max_len = max(max_len, j - i + 1)
        return max_len