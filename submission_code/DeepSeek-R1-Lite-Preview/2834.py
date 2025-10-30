from typing import List
from collections import defaultdict

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Step 1: Initialize the dictionary with the initial positions
        position_count = defaultdict(int)
        for pos in nums:
            position_count[pos] += 1
        
        # Step 2: Perform each move
        for frm, to in zip(moveFrom, moveTo):
            count = position_count[frm]
            position_count[to] += count
            del position_count[frm]
        
        # Step 3: Collect and sort the occupied positions
        final_positions = sorted(position_count.keys())
        return final_positions