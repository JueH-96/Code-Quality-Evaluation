from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        freq = Counter(nums)
        
        # If any frequency is 1, it's impossible to delete that element
        for count in freq.values():
            if count == 1:
                return -1
        
        operations = 0
        for count in freq.values():
            if count % 3 == 0:
                operations += count // 3
            elif count % 3 == 1:
                # Use (count//3 - 1) triplets and 2 pairs
                operations += (count // 3 - 1) + 2
            else:  # count % 3 == 2
                # Use count//3 triplets and 1 pair
                operations += (count // 3) + 1
        
        return operations