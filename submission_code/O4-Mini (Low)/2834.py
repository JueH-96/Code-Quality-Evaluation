from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Use a dictionary to track the count of marbles at each position
        count = {}
        for x in nums:
            count[x] = count.get(x, 0) + 1
        
        # Process each move
        for frm, to in zip(moveFrom, moveTo):
            if frm == to:
                # Moving to the same position does nothing
                continue
            # There is guaranteed to be at least one marble at frm
            c = count.pop(frm, 0)
            if c:
                count[to] = count.get(to, 0) + c
        
        # Return the sorted list of occupied positions
        return sorted(count.keys())