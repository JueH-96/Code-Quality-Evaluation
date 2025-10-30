from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # current occupied positions (uniquely)
        occupied = set(nums)
        
        # apply each move
        for src, dst in zip(moveFrom, moveTo):
            if src == dst:                 # moving to the same place changes nothing
                continue
            occupied.discard(src)          # guaranteed to exist; discard avoids KeyError
            occupied.add(dst)              # add the destination position
        
        # return the sorted list of occupied positions
        return sorted(occupied)