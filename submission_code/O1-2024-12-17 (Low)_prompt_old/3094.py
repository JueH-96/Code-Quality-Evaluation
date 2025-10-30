from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        freq = Counter(nums)
        operations = 0
        
        for count in freq.values():
            # If we have only 1 occurrence, we can't remove it by 2 or 3
            if count == 1:
                return -1
            
            # Greedy approach:
            # 1) use as many groups of 3 as possible
            # 2) then handle the remainder with groups of 2 if needed
            
            if count % 3 == 0:
                # Perfectly divisible by 3
                operations += count // 3
            elif count % 3 == 1:
                # If there are at least 4, we can form (count - 4) // 3 groups of 3
                # plus 2 groups of 2 (because 4 => 2 + 2)
                if count >= 4:
                    operations += (count - 4) // 3 + 2
                else:
                    return -1
            else:  # count % 3 == 2
                # (count // 3) groups of 3 plus 1 group of 2
                operations += count // 3 + 1
        
        return operations