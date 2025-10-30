class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
            
        max_strength = float('-inf')
        
        # Try all possible combinations using bit manipulation
        for mask in range(1, 1 << n):
            strength = 1
            for i in range(n):
                if mask & (1 << i):
                    strength *= nums[i]
            max_strength = max(max_strength, strength)
            
        return max_strength