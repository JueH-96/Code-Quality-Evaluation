from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0

        # Count the frequency of each number in the array
        freq = Counter(nums)

        # Count the number of operations
        operations = 0

        for count in freq.values():
            # If there's a single element left, it's impossible to remove it
            if count == 1:
                return -1

            # Calculate the number of operations needed for the current count
            if count % 3 == 0:
                operations += count // 3
            elif count % 3 == 1:
                # To handle the case where count % 3 == 1, we need to use two 2-operations and the rest 3-operations
                operations += (count - 4) // 3 + 2
            else:
                # For count % 3 == 2, use one 2-operation and the rest 3-operations
                operations += (count - 2) // 3 + 1

        return operations