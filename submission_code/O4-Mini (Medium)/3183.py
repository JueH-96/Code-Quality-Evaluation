from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # We'll count how many times each bit position is set among the numbers.
        # Since nums[i] < 2^31, we only need to check up to 31 bits (0 through 30).
        bit_counts = [0] * 31
        
        # Count bits
        for num in nums:
            for i in range(31):
                if (num >> i) & 1:
                    bit_counts[i] += 1
        
        # Build the K-or result: if a bit position i has at least k occurrences, set that bit.
        result = 0
        for i, cnt in enumerate(bit_counts):
            if cnt >= k:
                result |= (1 << i)
        
        return result