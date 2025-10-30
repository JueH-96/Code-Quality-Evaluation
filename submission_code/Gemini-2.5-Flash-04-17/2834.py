from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Use a set to keep track of occupied positions efficiently.
        # Sets provide efficient O(1) average time complexity for adding and removing elements,
        # which is suitable for handling potentially large and sparse position values.
        # Initial occupied positions are the unique values in nums.
        occupied_positions = set(nums)

        # Iterate through each move instruction.
        # For the i-th move, all marbles at position moveFrom[i] are moved to position moveTo[i].
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]

            # When all marbles move from `from_pos`, that position becomes empty.
            # We remove it from our set of occupied positions.
            # The problem guarantees that there is at least one marble at `from_pos`
            # at the time of the move, which implies `from_pos` is currently in the set.
            occupied_positions.remove(from_pos)

            # The marbles move to `to_pos`. This position is now occupied.
            # We add it to our set. If `to_pos` was already in the set (i.e., it was
            # already occupied by other marbles or a previous move), the set's
            # add operation handles this correctly by doing nothing.
            occupied_positions.add(to_pos)

        # After all moves are processed, the set `occupied_positions` contains
        # the final list of unique positions that hold at least one marble.
        # Convert the set to a list and sort it in ascending order as required by the output format.
        result = list(occupied_positions)
        result.sort() # Sorts the list in place

        return result