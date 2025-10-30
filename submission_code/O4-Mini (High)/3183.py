from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # There are at most 31 bits to consider (0 through 30)
        bit_counts = [0] * 31
        
        # Count how many numbers have each bit set
        for num in nums:
            for i in range(31):
                if (num >> i) & 1:
                    bit_counts[i] += 1
        
        # Build the K-or: set bit i if at least k numbers had it set
        result = 0
        for i, cnt in enumerate(bit_counts):
            if cnt >= k:
                result |= (1 << i)
        
        return result