from collections import Counter
from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        counts = Counter(nums)
        xor_sum = 0
        for num, count in counts.items():
            if count == 2:
                xor_sum ^= num
        return xor_sum