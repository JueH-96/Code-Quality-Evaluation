class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        max_strength = None
        n = len(nums)
        for i in range(1, 1 << n):
            product = 1
            for j in range(n):
                if (i >> j) & 1:
                    product *= nums[j]
            if max_strength is None or product > max_strength:
                max_strength = product
        return max_strength