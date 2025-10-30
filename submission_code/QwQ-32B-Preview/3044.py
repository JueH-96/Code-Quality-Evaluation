from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        required = set(range(1, k+1))
        collected = set()
        operations = 0
        # Iterate from the end of the array
        for i in range(len(nums)-1, -1, -1):
            operations += 1
            if nums[i] in required:
                collected.add(nums[i])
                if collected.issuperset(required):
                    return operations
        return operations  # This line is theoretically unreachable due to constraints