from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        # We iterate over 31 bits because each number is strictly less than 2^31.
        for bit in range(31):
            mask = 1 << bit
            # Count how many numbers have the current bit set
            count = sum(1 for num in nums if num & mask)
            # If at least k numbers have this bit set, include it in result.
            if count >= k:
                result |= mask
        return result