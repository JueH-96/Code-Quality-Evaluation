from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # This solution maintains sets of possible OR results ending at every index.
        cur = set()
        best = float('inf')
        
        for num in nums:
            new_set = {num}
            # For each OR value ending at previous index, extend the subarray with num.
            for prev in cur:
                new_set.add(prev | num)
                
            # Update our best difference.
            for val in new_set:
                best = min(best, abs(k - val))
                # Early return if perfect match is found
                if best == 0:
                    return 0
            
            cur = new_set
        
        return best