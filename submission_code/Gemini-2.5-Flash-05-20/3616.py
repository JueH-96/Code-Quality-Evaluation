import collections
from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        valid_selections_count = 0

        # Define direction constants for readability
        LEFT = -1
        RIGHT = 1

        # Iterate through all possible starting positions
        # A starting position 'curr' must have nums[curr] == 0
        for start_idx in range(n):
            if nums[start_idx] == 0:
                # For each valid starting position, try both initial directions
                for initial_direction in [LEFT, RIGHT]:
                    # Create a deep copy of the original nums array for the current simulation.
                    # This is crucial because the simulation modifies the array.
                    # list[:] creates a shallow copy, which is sufficient for a list of integers.
                    curr_nums = nums[:] 
                    curr_pos = start_idx
                    curr_dir = initial_direction
                    
                    # Simulate the movement process
                    # The loop continues as long as 'curr_pos' is within the array bounds.
                    # The process is guaranteed to terminate because:
                    # 1. If curr_nums[curr_pos] is consistently 0, curr_pos will eventually move out of bounds.
                    # 2. If curr_nums[curr_pos] is positive, it decrements, contributing to the total sum decreasing.
                    #    Since the sum of positive numbers eventually becomes zero, and this involves movement,
                    #    curr_pos will eventually move out of bounds.
                    while 0 <= curr_pos < n:
                        if curr_nums[curr_pos] == 0:
                            # If the current cell is 0, just move in the current direction
                            curr_pos += curr_dir
                        else: # curr_nums[curr_pos] > 0
                            # If the current cell is positive:
                            # 1. Decrement its value
                            curr_nums[curr_pos] -= 1
                            # 2. Reverse the movement direction
                            curr_dir *= -1 
                            # 3. Take a step in the new direction
                            curr_pos += curr_dir
                    
                    # After the process ends (curr_pos is out of bounds),
                    # check if all elements in the array have become zero.
                    all_zeros = True
                    for x in curr_nums:
                        if x != 0:
                            all_zeros = False
                            break
                    
                    # If all elements are zero, this selection is valid
                    if all_zeros:
                        valid_selections_count += 1
                        
        return valid_selections_count