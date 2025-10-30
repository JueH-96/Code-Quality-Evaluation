from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        count = Counter(nums)
        for freq in count.values():
            if freq > 2:
                return False
        m = len(count)
        k = len(nums) // 2
        return m >= k