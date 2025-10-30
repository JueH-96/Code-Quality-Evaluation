class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        n = len(nums)
        
        # Check all possible pairs
        for i in range(n):
            for j in range(i, n):  # j starts from i to include same element pairs
                x, y = nums[i], nums[j]
                
                # Check if it's a strong pair
                if abs(x - y) <= min(x, y):
                    # Calculate XOR and update maximum
                    max_xor = max(max_xor, x ^ y)
        
        return max_xor