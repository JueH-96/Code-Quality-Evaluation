from typing import List
from collections import Counter

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        count = Counter(nums)
        result = 0
        for num, cnt in count.items():
            if cnt == 2:
                result ^= num
        return result