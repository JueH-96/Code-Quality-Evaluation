from typing import List
from collections import Counter

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Initialize a counter to keep track of marble positions
        marble_positions = Counter(nums)

        # Process each move
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]

            # If there are marbles at the from_pos, move them to to_pos
            if marble_positions[from_pos] > 0:
                marble_positions[to_pos] += marble_positions[from_pos]
                del marble_positions[from_pos]

        # Extract the positions that have marbles and sort them
        occupied_positions = sorted(marble_positions.keys())

        return occupied_positions