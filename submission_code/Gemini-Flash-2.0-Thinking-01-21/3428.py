from collections import Counter
from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        counts = Counter(nums)
        duplicate_numbers = []
        for num, count in counts.items():
            if count == 2:
                duplicate_numbers.append(num)
        
        xor_sum = 0
        for num in duplicate_numbers:
            xor_sum ^= num
            
        return xor_sum