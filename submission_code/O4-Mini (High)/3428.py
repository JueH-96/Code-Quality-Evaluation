from typing import List
from collections import Counter

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        freq = Counter(nums)
        res = 0
        for num, cnt in freq.items():
            if cnt == 2:
                res ^= num
        return res