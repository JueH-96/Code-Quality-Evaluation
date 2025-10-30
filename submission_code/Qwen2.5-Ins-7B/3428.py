from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        xor_result = 0
        for num in nums:
            if nums.count(num) == 2:
                xor_result ^= num
        return xor_result