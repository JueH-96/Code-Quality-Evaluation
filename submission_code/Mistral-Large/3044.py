from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # We need to collect elements 1 through k
        needed = set(range(1, k + 1))
        operations = 0

        # Iterate over the array in reverse order
        for num in reversed(nums):
            operations += 1
            needed.discard(num)
            # If we have collected all needed elements, we can stop
            if not needed:
                break

        return operations