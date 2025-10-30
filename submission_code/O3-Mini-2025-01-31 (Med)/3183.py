from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        result = 0
        # We iterate over all possible bit positions from 0 to 30 (inclusive)
        # as numbers can be up to 2^31 - 1.
        for i in range(31):
            count = 0
            # Count how many numbers have the i-th bit set.
            for num in nums:
                if num & (1 << i):
                    count += 1
            # If at least k numbers have the i-th bit set, set that bit in the result.
            if count >= k:
                result |= (1 << i)
        return result