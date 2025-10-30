from typing import List
from collections import Counter

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        # Count frequency of each number in the list
        freq = Counter(nums)
        
        # Compute the XOR for numbers that appear exactly twice
        result = 0
        for num, count in freq.items():
            if count == 2:
                result ^= num
                
        return result