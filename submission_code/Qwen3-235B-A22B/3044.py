from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        required = set(range(1, k + 1))
        collected = set()
        count = 0
        for num in reversed(nums):
            collected.add(num)
            count += 1
            if required.issubset(collected):
                return count
        return count