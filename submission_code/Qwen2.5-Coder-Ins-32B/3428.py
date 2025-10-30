from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        seen = set()
        xor_result = 0
        for num in nums:
            if num in seen:
                xor_result ^= num
            else:
                seen.add(num)
        return xor_result