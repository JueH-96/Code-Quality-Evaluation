from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        duplicates = [k for k, v in freq.items() if v == 2]
        result = 0
        for num in duplicates:
            result ^= num
        return result