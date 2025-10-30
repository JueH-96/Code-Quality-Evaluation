from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Build a dictionary counting how many marbles are at each position initially.
        marble_positions = {}
        for pos in nums:
            marble_positions[pos] = marble_positions.get(pos, 0) + 1

        # Process each move: move all marbles from moveFrom[i] to moveTo[i]
        for src, dest in zip(moveFrom, moveTo):
            # Remove all marbles from the source position
            count = marble_positions.pop(src)
            # Add these marbles to the destination position (may already have some)
            marble_positions[dest] = marble_positions.get(dest, 0) + count

        # Return the sorted list of positions that have at least one marble
        return sorted(marble_positions.keys())