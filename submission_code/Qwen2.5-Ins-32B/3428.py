from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        from collections import Counter
        count = Counter(nums)
        xor_result = 0
        for num, freq in count.items():
            if freq == 2:
                xor_result ^= num
        return xor_result