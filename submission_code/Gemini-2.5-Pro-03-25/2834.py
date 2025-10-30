import collections # This import is not strictly necessary for the final solution but might be useful for alternative approaches or debugging. It's generally safe to keep standard library imports if they were part of the thought process, but can be removed if unused. Let's remove it for cleanliness.
from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        """
        Simulates the movement of marbles based on a series of moves specified by 
        moveFrom and moveTo arrays, starting from initial positions in nums.
        Returns the sorted list of final positions that are occupied by at least one marble.

        Args:
            nums: A list of integers representing the initial positions of marbles. 
                  There might be multiple marbles at the same position.
            moveFrom: A list of integers representing the source positions for moves.
                      The i-th move involves moving marbles from moveFrom[i].
            moveTo: A list of integers representing the destination positions for moves.
                    The i-th move involves moving marbles to moveTo[i].
                    moveFrom and moveTo have the same length.

        Returns:
            A sorted list of integers representing the final unique positions 
            occupied by at least one marble after all moves are completed.
        """
        
        # Step 1: Initialize a set with the unique initial positions of the marbles.
        # Using a set automatically handles duplicate initial positions in `nums` 
        # and provides efficient O(1) average time complexity for add, remove, 
        # and membership check operations. This set will track the positions 
        # that are currently occupied.
        occupied_positions = set(nums)
        
        # Step 2: Iterate through each move step.
        # The `zip` function pairs corresponding elements from `moveFrom` and `moveTo`,
        # allowing us to process each move (source, destination) together.
        for source, destination in zip(moveFrom, moveTo):
            
            # The problem statement provides a crucial constraint: 
            # "The test cases are generated such that there is at least a marble 
            # in moveFrom[i] at the moment we want to apply the i^th move."
            # This guarantees that the `source` position is present in our 
            # `occupied_positions` set just before we execute the move.
            # Therefore, we don't need an explicit check like `if source in occupied_positions:`.
            
            # Perform the move simulation by updating the set of occupied positions:
            
            # 1. Remove the source position from the set. 
            #    Since the constraint guarantees `source` is in the set, 
            #    this `remove` operation will always succeed without raising a KeyError.
            occupied_positions.remove(source)
            
            # 2. Add the destination position to the set.
            #    - If the `destination` position was already occupied (i.e., already in the set), 
            #      the set remains unchanged regarding this element (sets only store unique items).
            #    - If `source` and `destination` were the same, the `remove` operation followed 
            #      by this `add` operation correctly results in the position remaining in the set, 
            #      indicating it stays occupied.
            occupied_positions.add(destination)
            
        # Step 3: After processing all the moves, the `occupied_positions` set contains 
        # all the final positions where at least one marble resides.
        
        # Step 4: Convert the set of final occupied positions into a list.
        final_positions_list = list(occupied_positions)
        
        # Step 5: Sort the list in ascending order as required by the problem output format.
        # The `sorted()` function returns a new sorted list based on the elements 
        # of the iterable provided (in this case, the list derived from the set).
        sorted_final_positions = sorted(final_positions_list)
        
        # Step 6: Return the sorted list.
        return sorted_final_positions