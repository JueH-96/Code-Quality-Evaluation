from collections import Counter
from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        count = Counter(nums)
        duplicates = [num for num, freq in count.items() if freq == 2]
        result = 0
        for num in duplicates:
            result ^= num
        return result