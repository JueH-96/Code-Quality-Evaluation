from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        xor_val = 0
        for num, count in freq.items():
            if count == 2:
                xor_val ^= num
                
        return xor_val