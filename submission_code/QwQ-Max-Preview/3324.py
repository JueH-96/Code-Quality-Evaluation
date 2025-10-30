from collections import Counter
from typing import List

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        freq = Counter(nums)
        # Check if any element occurs more than twice
        for count in freq.values():
            if count > 2:
                return False
        # Count the number of elements that appear exactly twice
        p = sum(1 for count in freq.values() if count == 2)
        k = len(nums) // 2
        return p <= k