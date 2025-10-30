from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Step 1: Initialize a set with all initial occupied positions.
        # Using a set is efficient for managing unique positions, providing O(1) average
        # time complexity for additions, removals, and lookups.
        occupied_positions = set(nums)

        # Step 2: Process each movement.
        # Iterate through the moveFrom and moveTo arrays in parallel.
        for i in range(len(moveFrom)):
            source_pos = moveFrom[i]
            destination_pos = moveTo[i]

            # Remove the source position from the set of occupied positions.
            # The problem guarantees that `source_pos` will always be occupied
            # when we attempt to move marbles from it, so no prior check is needed.
            occupied_positions.remove(source_pos)
            
            # Add the destination position to the set of occupied positions.
            # If `destination_pos` was already in the set, adding it again has no effect,
            # which is the correct behavior as the position remains occupied.
            occupied_positions.add(destination_pos)
        
        # Step 3: Convert the set of final occupied positions to a list.
        result = list(occupied_positions)
        
        # Step 4: Sort the list in ascending order as required by the problem.
        result.sort()
        
        # Step 5: Return the sorted list of occupied positions.
        return result