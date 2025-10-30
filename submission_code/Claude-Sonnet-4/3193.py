class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        n = len(nums)
        
        # Check all possible pairs
        for i in range(n):
            for j in range(i, n):
                x, y = nums[i], nums[j]
                
                # Check if it's a strong pair: |x - y| <= min(x, y)
                if abs(x - y) <= min(x, y):
                    # Calculate XOR and update maximum
                    current_xor = x ^ y
                    max_xor = max(max_xor, current_xor)
        
        return max_xor