from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Count the frequency of each number in the array
        freq = Counter(nums)
        operations = 0

        for count in freq.values():
            if count == 1:
                # If there is only one occurrence of a number, it's impossible to remove it
                return -1
            # For each count, we need to determine the minimum number of operations
            # to remove all occurrences of that number
            while count > 0:
                if count % 3 == 0:
                    # If count is divisible by 3, we can remove all occurrences in one operation
                    operations += count // 3
                    count = 0
                elif count % 3 == 1:
                    # If count leaves a remainder of 1 when divided by 3, we need to remove 4 numbers
                    # (one operation of 3 and one operation of 1)
                    operations += (count // 3) + 1
                    count = 1
                else:
                    # If count leaves a remainder of 2 when divided by 3, we need to remove 5 numbers
                    # (one operation of 3 and one operation of 2)
                    operations += (count // 3) + 1
                    count = 2

        return operations