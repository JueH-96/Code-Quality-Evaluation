import bisect
from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        """
        Counts the number of valid starting selections (position and direction)
        that result in all elements of nums becoming 0.
        """
        n = len(nums)
        # Find all possible starting positions (where nums[i] == 0)
        zero_indices = [i for i, val in enumerate(nums) if val == 0]
        
        valid_selections_count = 0
        
        # A movement direction can be right (1) or left (-1)
        directions = [1, -1]
        
        # Iterate over all possible starting selections
        for start_pos in zero_indices:
            for start_dir in directions:
                # Run a simulation for each selection and check if it's valid
                if self._run_simulation(nums, n, start_pos, start_dir):
                    valid_selections_count += 1
                    
        return valid_selections_count

    def _run_simulation(self, original_nums: List[int], n: int, start_pos: int, start_dir: int) -> bool:
        """
        Runs a full simulation for a given starting selection.
        Returns True if the selection is valid, False otherwise.
        """
        # Create a copy of the array for this simulation
        nums = list(original_nums)
        # Create a sorted list of indices with non-zero values for efficient lookups
        nonzero_indices = sorted([i for i, v in enumerate(nums) if v > 0])
        
        curr = start_pos
        direction = start_dir
        
        while 0 <= curr < n:
            if nums[curr] > 0:
                # Non-zero case: decrement, reverse direction, and take a step
                nums[curr] -= 1
                
                if nums[curr] == 0:
                    # If the value becomes 0, remove its index from our tracking list
                    idx = bisect.bisect_left(nonzero_indices, curr)
                    nonzero_indices.pop(idx)
                
                direction *= -1
                curr += direction
            else:  # nums[curr] == 0
                # Zero case: slide/jump to the next non-zero element
                if direction == 1:  # Moving right
                    # Find the first non-zero index > curr
                    i = bisect.bisect_right(nonzero_indices, curr)
                    if i == len(nonzero_indices):
                        curr = n  # Moved past the right end
                    else:
                        curr = nonzero_indices[i]
                else:  # Moving left (direction == -1)
                    # Find the first non-zero index < curr
                    i = bisect.bisect_left(nonzero_indices, curr)
                    if i == 0:
                        curr = -1  # Moved past the left end
                    else:
                        curr = nonzero_indices[i - 1]
                        
        # The simulation ends when curr goes out of bounds.
        # A run is valid if all numbers are now zero, which means our
        # list of non-zero indices is empty.
        return len(nonzero_indices) == 0