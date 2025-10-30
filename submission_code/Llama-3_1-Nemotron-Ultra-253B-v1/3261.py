from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        m = len(nums) - k
        result = 0
        for b in reversed(range(31)):  # Check each bit from highest to lowest
            mask = 1 << b
            count = 0
            for num in nums:
                if (num & mask) == 0:
                    count += 1
                    if count >= m:
                        break
            if count < m:
                result |= mask
        return result