class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_product = -10**18
        for mask in range(1, 1 << n):
            product = 1
            for i in range(n):
                if mask & (1 << i):
                    product *= nums[i]
            if product > max_product:
                max_product = product
        return max_product