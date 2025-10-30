from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Count frequency of each number
        freq = Counter(nums)
        total_ops = 0
        
        # Process each frequency separately
        for count in freq.values():
            # If a number appears only once, it is impossible to remove it using operations of 2 or 3.
            if count == 1:
                return -1
            
            # We want to partition count = 2*x + 3*y, minimizing (x+y).
            # A greedy strategy is to use as many triple removals as possible since 3-item removal
            # clears more items per operation, but we must adjust for remainders.
            y = count // 3
            remainder = count % 3
            
            if remainder == 0:
                total_ops += y
            elif remainder == 1:
                # For remainder 1, we can convert one triple (if available) into two pairs.
                # For example, count = 4 (1 triple removed -> leaves 1, not valid) but then 0 triple + 2 pairs = 2 ops.
                # So if y >= 1, adjust accordingly.
                if y >= 1:
                    y -= 1  # use one less triple removal
                    # The removed triple gives back 3 items. Add the leftover remainder 1 to have 4.
                    # 4 can be removed as two pairs.
                    total_ops += y + 2
                else:
                    # If not possible to adjust (which happens when count=1, but we've already returned for count==1),
                    # we return -1 for safety.
                    return -1
            elif remainder == 2:
                total_ops += y + 1
        
        return total_ops