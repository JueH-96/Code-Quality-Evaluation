from collections import Counter
from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        freq = Counter(nums)
        duplicates = [num for num, count in freq.items() if count == 2]
        result = 0
        for num in duplicates:
            result ^= num
        return result