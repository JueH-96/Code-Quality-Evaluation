class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        best = float('-inf')
        
        # Iterate through all non-empty subsets using a bitmask approach
        for mask in range(1, 1 << n):
            product = 1
            for i in range(n):
                if mask & (1 << i):
                    product *= nums[i]
            best = max(best, product)
        
        return best