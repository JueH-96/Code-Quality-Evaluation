from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        # Helper to compute LCM of two positive ints
        def lcm(a: int, b: int) -> int:
            return a // math.gcd(a, b) * b

        n = len(nums)
        # Special case: single element
        if n == 1:
            return nums[0] * nums[0]

        # Build prefix GCD and LCM
        prefix_gcd = [0] * n
        prefix_lcm = [0] * n
        prefix_gcd[0] = nums[0]
        prefix_lcm[0] = nums[0]
        for i in range(1, n):
            prefix_gcd[i] = math.gcd(prefix_gcd[i-1], nums[i])
            prefix_lcm[i] = lcm(prefix_lcm[i-1], nums[i])

        # Build suffix GCD and LCM
        suffix_gcd = [0] * n
        suffix_lcm = [0] * n
        suffix_gcd[n-1] = nums[n-1]
        suffix_lcm[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix_gcd[i] = math.gcd(suffix_gcd[i+1], nums[i])
            suffix_lcm[i] = lcm(suffix_lcm[i+1], nums[i])

        # Option 1: remove no element
        best = prefix_gcd[n-1] * prefix_lcm[n-1]

        # Option 2: remove one element at index i
        for i in range(n):
            if i == 0:
                g = suffix_gcd[1]
                L = suffix_lcm[1]
            elif i == n-1:
                g = prefix_gcd[n-2]
                L = prefix_lcm[n-2]
            else:
                g = math.gcd(prefix_gcd[i-1], suffix_gcd[i+1])
                L = lcm(prefix_lcm[i-1], suffix_lcm[i+1])
            best = max(best, g * L)

        return best