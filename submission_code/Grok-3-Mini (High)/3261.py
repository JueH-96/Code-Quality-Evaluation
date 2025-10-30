from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n - k
        or_value = 0
        for bit in range(30):
            mask = 1 << bit
            unset_count = sum(1 for num in nums if (num & mask) == 0)
            if unset_count < m:
                or_value |= (1 << bit)
        return or_value