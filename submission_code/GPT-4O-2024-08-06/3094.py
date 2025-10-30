from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Count the frequency of each number in the array
        freq = Counter(nums)
        
        operations = 0
        
        for count in freq.values():
            # If the count of any number is 1, it's impossible to remove it
            if count == 1:
                return -1
            
            # Calculate the minimum operations needed for this count
            # We can use either pairs or triplets
            # If count is divisible by 3, we can use only triplets
            # If count % 3 == 1, we need to use one pair and the rest triplets
            # If count % 3 == 2, we can use one pair and the rest triplets
            if count % 3 == 0:
                operations += count // 3
            elif count % 3 == 1:
                operations += (count // 3) - 1 + 2  # Use one less triplet and two pairs
            elif count % 3 == 2:
                operations += (count // 3) + 1  # Use one pair and the rest triplets
        
        return operations