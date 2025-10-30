from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Step 1: Initialize a set with the initial positions of marbles.
        # Using a set automatically handles duplicate positions in nums
        # and provides efficient add/remove operations.
        occupied_positions = set(nums)
        
        # Step 2: Process each move.
        # Iterate through each move specified by moveFrom and moveTo arrays.
        for i in range(len(moveFrom)):
            source_pos = moveFrom[i]
            target_pos = moveTo[i]
            
            # The problem guarantees that source_pos is currently occupied.
            # "The test cases are generated such that there is at least a marble
            # in moveFrom[i] at the moment we want to apply the i^th move."
            # This means source_pos will be in occupied_positions, so remove() will not fail.
            
            # If source_pos and target_pos are the same, these operations
            # correctly result in the position remaining in the set (it's removed then added back).
            occupied_positions.remove(source_pos)
            occupied_positions.add(target_pos)
            
        # Step 3: Convert the set of final occupied positions to a sorted list.
        # The sorted() function can take any iterable (like a set) and returns a new sorted list.
        final_sorted_positions = sorted(list(occupied_positions)) # or simply sorted(occupied_positions)
        
        return final_sorted_positions