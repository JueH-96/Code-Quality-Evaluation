class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_strength = float('-inf')
        
        # Try all possible non-empty subsets using bit manipulation
        for mask in range(1, 1 << n):  # from 1 to 2^n - 1
            product = 1
            for i in range(n):
                if mask & (1 << i):  # if i-th bit is set
                    product *= nums[i]
            max_strength = max(max_strength, product)
        
        return max_strength