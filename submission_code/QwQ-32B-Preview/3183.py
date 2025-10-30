from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # Initialize a list to count set bits for each position (0 to 31)
        bit_counts = [0] * 32
        
        # Count set bits for each number in nums
        for num in nums:
            for i in range(32):
                if (num >> i) & 1:
                    bit_counts[i] += 1
        
        # Build the result by setting bits where count >= k
        result = 0
        for i in range(32):
            if bit_counts[i] >= k:
                result |= (1 << i)
        
        return result