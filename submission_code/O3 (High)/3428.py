from typing import List
from collections import Counter

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        """
        Computes the bitwise XOR of all numbers that appear exactly twice in the list.
        If no number appears twice, returns 0.
        """
        freq = Counter(nums)           # Count occurrences of each number
        ans = 0
        
        for num, cnt in freq.items():  # Iterate through frequency table
            if cnt == 2:               # Consider only numbers that appear twice
                ans ^= num             # XOR them together
                
        return ans