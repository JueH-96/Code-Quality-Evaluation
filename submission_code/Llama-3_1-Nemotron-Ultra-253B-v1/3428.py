from typing import List
from collections import Counter

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        counts = Counter(nums)
        result = 0
        for num, cnt in counts.items():
            if cnt == 2:
                result ^= num
        return result