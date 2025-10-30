from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        total = 0
        for n in nums:
            rem = n % 3
            if rem != 0:
                total += min(rem, 3 - rem)
        return total