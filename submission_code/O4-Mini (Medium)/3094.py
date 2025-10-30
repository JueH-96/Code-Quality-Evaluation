from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        freq = Counter(nums)
        ops = 0
        
        for count in freq.values():
            # If there's only one occurrence, we can't pair or triple it
            if count == 1:
                return -1
            
            # If count is divisible by 3, use all triples
            if count % 3 == 0:
                ops += count // 3
            # If remainder is 1, better to take two pairs (4 items) and the rest triples
            elif count % 3 == 1:
                # since count >= 4 (we already ruled out count == 1),
                # we can subtract 4 and form two pairs, the rest in triples
                ops += (count - 4) // 3 + 2
            # If remainder is 2, take one pair and the rest triples
            else:  # count % 3 == 2
                ops += (count - 2) // 3 + 1
        
        return ops