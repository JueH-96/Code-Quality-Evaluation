from typing import List
from collections import Counter

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        count = Counter(nums)
        xor_result = 0
        for num in count:
            if count[num] == 2:
                xor_result ^= num
        return xor_result