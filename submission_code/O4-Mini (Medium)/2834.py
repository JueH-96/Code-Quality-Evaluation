from typing import List
import collections

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Count how many marbles are at each position
        counts = collections.Counter(nums)
        
        # Perform each move
        for src, dst in zip(moveFrom, moveTo):
            if src in counts:
                counts[dst] = counts.get(dst, 0) + counts[src]
                del counts[src]
        
        # Return the sorted list of occupied positions
        return sorted(counts.keys())