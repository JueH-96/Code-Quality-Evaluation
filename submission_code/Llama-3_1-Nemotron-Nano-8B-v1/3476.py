from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        total = 0
        for num in nums:
            rem = num % 3
            if rem == 0:
                continue
            total += min(rem, 3 - rem)
        return total