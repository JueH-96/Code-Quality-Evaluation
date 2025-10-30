from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        freq = {}
        xor_result = 0
        for num in nums:
            if num in freq:
                freq[num] += 1
                xor_result ^= num
            else:
                freq[num] = 1
        return xor_result