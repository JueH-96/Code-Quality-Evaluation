class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_strength = float("-inf")
        
        # Iterate over all possible non-empty subsets using bitmask
        # (there are 2^n subsets, excluding the empty one)
        for mask in range(1, 1 << n):
            product = 1
            for i in range(n):
                if mask & (1 << i):
                    product *= nums[i]
            max_strength = max(max_strength, product)
        
        return max_strength