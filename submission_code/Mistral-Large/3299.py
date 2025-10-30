from typing import List
from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count the occurrences of each number in the array
        count = Counter(nums)

        max_length = 0

        # Check each number as the base x
        for x in count:
            k = 0
            length = 1
            while x ** (2 ** k) in count:
                length += 2
                k += 1
            # Update the maximum length found
            max_length = max(max_length, length)

        return max_length