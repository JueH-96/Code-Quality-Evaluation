from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # We will count, for each bit position i, how many numbers have that bit set.
        # If the count >= k, we set bit i in the result.
        result = 0
        
        # Since nums[i] < 2^31, we only need to consider bits 0 through 30 inclusive.
        for bit in range(31):
            mask = 1 << bit
            count = 0
            # Count how many numbers have this bit set
            for num in nums:
                if num & mask:
                    count += 1
                    # Early exit if we've already reached k
                    if count >= k:
                        result |= mask
                        break
        
        return result