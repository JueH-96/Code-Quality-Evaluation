class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_val = -10**18
        
        for mask in range(1, 1 << n):
            product = 1
            for i in range(n):
                if mask & (1 << i):
                    product *= nums[i]
            if product > max_val:
                max_val = product
                
        return max_val