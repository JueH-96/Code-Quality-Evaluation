from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        """
        Simulates the relocation of marbles and returns the final occupied positions.
        """
        
        # Use a set to efficiently track the unique occupied positions.
        # Initializing from `nums` automatically handles duplicates and sets up the start state.
        positions = set(nums)
        
        # Process each move. zip() is used to iterate over source and destination pairs.
        for src, dest in zip(moveFrom, moveTo):
            # The problem guarantees that `src` is an occupied position at the time of the move.
            # So we can safely remove the source position.
            positions.remove(src)
            
            # Add the destination position to the set of occupied positions.
            # If it's already present, the set remains unchanged.
            positions.add(dest)
            
        # The result must be a sorted list of the final occupied positions.
        # sorted() can be called directly on a set to produce a sorted list.
        return sorted(positions)