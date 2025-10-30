from collections import Counter
from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        counts = Counter(nums)
        xor_val = 0
        for num in counts:
            if counts[num] == 2:
                xor_val ^= num
        return xor_val