from typing import List

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        n = len(nums)
        
        # Generate all possible pairs of numbers
        for i in range(n):
            for j in range(n):
                # Check if the pair is a strong pair
                if abs(nums[i] - nums[j]) <= min(nums[i], nums[j]):
                    # Calculate the XOR of the pair
                    xor = nums[i] ^ nums[j]
                    # Update the maximum XOR
                    max_xor = max(max_xor, xor)
        
        return max_xor