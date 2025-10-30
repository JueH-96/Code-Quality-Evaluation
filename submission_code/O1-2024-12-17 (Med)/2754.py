class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        best = float('-inf')
        # Enumerate all non-empty subsets using bitmasks
        for mask in range(1, 1 << n):
            product_val = 1
            for i in range(n):
                if mask & (1 << i):
                    product_val *= nums[i]
            best = max(best, product_val)
        return best