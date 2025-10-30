from typing import List
from collections import Counter

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        counts = Counter(nums)
        duplicates = [num for num, cnt in counts.items() if cnt == 2]
        if not duplicates:
            return 0
        result = 0
        for num in duplicates:
            result ^= num
        return result