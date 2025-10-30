from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        operations = 0
        
        for freq in count.values():
            if freq % 2 == 1:
                return -1  # If any frequency is odd, we cannot pair them all
            
            operations += freq // 3  # Count how many triplets we can form
            freq %= 3  # Remaining elements after forming triplets
            
            operations += freq // 2  # Count how many pairs we can form
        
        return operations