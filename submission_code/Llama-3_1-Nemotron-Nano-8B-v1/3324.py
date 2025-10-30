from collections import Counter
from typing import List

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        count = Counter(nums)
        # Check if any element occurs more than twice
        for key in count:
            if count[key] > 2:
                return False
        # Count the number of elements that occur exactly twice
        k = sum(1 for key in count if count[key] == 2)
        n = len(nums)
        # Check if the number of elements with two occurrences is at most half the array length
        return k <= n // 2