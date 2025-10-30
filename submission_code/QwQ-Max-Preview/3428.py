from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        freq = {}
        for num in nums:
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
        
        xor_result = 0
        for num, count in freq.items():
            if count == 2:
                xor_result ^= num
        
        return xor_result