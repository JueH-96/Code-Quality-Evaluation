from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        freq = Counter(nums)
        operations = 0
        
        for val, count in freq.items():
            if count == 1:
                # Can't remove a single element
                return -1
            
            # Divide count by 3, check remainder
            k, r = divmod(count, 3)
            if r == 0:
                # All triples
                operations += k
            elif r == 1:
                # One leftover: we must break one triple into two pairs (if possible)
                # That costs (k-1) triples + 2 pairs = (k-1) + 2 ops = k + 1
                if k >= 1:
                    operations += (k - 1) + 2
                else:
                    # Means count == 1 which is impossible (we already handled), but guard
                    return -1
            else:  # r == 2
                # k triples + one pair
                operations += k + 1
        
        return operations