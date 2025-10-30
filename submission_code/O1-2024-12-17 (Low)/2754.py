class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_product = -float('inf')
        
        # Iterate over all non-empty subsets using bitmasks
        for mask in range(1, 1 << n):
            product = 1
            for i in range(n):
                if (mask >> i) & 1:  # if the i-th bit of mask is set
                    product *= nums[i]
            max_product = max(max_product, product)
        
        return max_product