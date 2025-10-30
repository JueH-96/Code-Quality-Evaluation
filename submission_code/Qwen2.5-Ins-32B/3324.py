from typing import List

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        count = Counter(nums)
        for val in count.values():
            if val > 2:
                return False
        return True