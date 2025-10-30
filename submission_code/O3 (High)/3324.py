from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        # For every value there can be at most one copy in each of the two parts,
        # so a value occurring three or more times makes a valid split impossible.
        for freq in Counter(nums).values():
            if freq > 2:
                return False
        return True