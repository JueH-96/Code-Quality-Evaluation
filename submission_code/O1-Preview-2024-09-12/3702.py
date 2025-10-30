class Solution:
    def maxLength(self, nums: List[int]) -> int:
        import math

        def lcm(a, b):
            return abs(a * b) // math.gcd(a, b)

        max_len = 0
        n = len(nums)
        for i in range(n):
            prod = nums[i]
            curr_gcd = nums[i]
            curr_lcm = nums[i]
            if prod == curr_gcd * curr_lcm:
                max_len = max(max_len, 1)
            for j in range(i + 1, n):
                prod *= nums[j]
                curr_gcd = math.gcd(curr_gcd, nums[j])
                curr_lcm = lcm(curr_lcm, nums[j])
                if prod == curr_gcd * curr_lcm:
                    max_len = max(max_len, j - i + 1)
        return max_len