from typing import List
import collections

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        freq = collections.Counter(nums)
        result = 0
        for num, count in freq.items():
            if count == 2:
                result ^= num
        return result