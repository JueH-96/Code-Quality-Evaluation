from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = Counter(nums)
        total_operations = 0

        for count in counts.values():
            if count == 1:
                return -1
            elif count % 3 == 0:
                total_operations += count // 3
            elif count % 3 == 1:
                # We need to use pairs of 2 to reduce the remainder to 0
                # count = 3k + 1
                # count - 4 = 3k - 3 = 3(k - 1)
                total_operations += (count - 4) // 3 + 2
            elif count % 3 == 2:
                # We need to use one pair of 2 to reduce the remainder to 0
                # count = 3k + 2
                # count - 2 = 3k
                total_operations += (count - 2) // 3 + 1

        return total_operations