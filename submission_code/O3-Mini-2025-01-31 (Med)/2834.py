from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Create a dictionary to count marbles at each position
        marble_count = {}
        for pos in nums:
            marble_count[pos] = marble_count.get(pos, 0) + 1
        
        # Process each move from moveFrom[i] to moveTo[i]
        for frm, to in zip(moveFrom, moveTo):
            # Get the count of marbles to move
            count = marble_count.get(frm, 0)
            # As per the problem constraints, count > 0 always here.
            # Remove these marbles from the 'from' position
            del marble_count[frm]
            # Add these marbles to the 'to' position
            marble_count[to] = marble_count.get(to, 0) + count
        
        # Return the sorted list of positions that remain occupied (i.e., have at least one marble)
        return sorted(marble_count.keys())