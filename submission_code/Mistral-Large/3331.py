from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        operations = 0

        for num in nums:
            if num >= k:
                break
            operations += 1

        return operations