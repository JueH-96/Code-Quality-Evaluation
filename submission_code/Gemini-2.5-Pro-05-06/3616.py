from typing import List

class Solution:
  def countValidSelections(self, nums: List[int]) -> int:
    n = len(nums)
    initial_zeros_indices = []
    for i, val in enumerate(nums):
        if val == 0:
            initial_zeros_indices.append(i)
    
    valid_selections_count = 0
    
    # Calculate s_initial = sum(nums) once.
    # This is used for calculating current_s efficiently and for the step limit.
    s_initial = 0
    for x in nums:
        s_initial += x

    # Optimization: If all numbers are initially zero.
    if s_initial == 0:
        # Any choice of starting zero and direction is valid.
        # The simulation will just move curr until it goes out of bounds.
        # All numbers in the array remain zero throughout this process.
        return len(initial_zeros_indices) * 2

    # Max iterations for the simulation loop.
    # A single simulation involves S_initial interactions (decrementing non-zero cells).
    # This divides the process into S_initial + 1 phases of movement in a fixed direction.
    # Each phase can traverse up to N zero cells.
    # So, number of times `curr` is updated is bounded by (S_initial + 1) * N.
    # Add a small constant (+1) for safety margin or edge cases like N=0 (not possible by constraints) or S_initial=0 (handled).
    # Example: nums=[0,0,0], N=3, S_initial=0. max_loop_iterations = (0+1)*3 + 1 = 4.
    # A path like 0->1->2->3(OOB) takes 3 iterations. The bound N+1 is safe.
    max_loop_iterations = (s_initial + 1) * n + 1 # Max number of times the simulation step can run
                                                
    for start_idx in initial_zeros_indices:
        # Directions: -1 for left, 1 for right.
        for initial_direction_val in [-1, 1]: 
            
            current_nums = list(nums) # Create a mutable copy of nums for this simulation.
            curr = start_idx
            direction = initial_direction_val
            
            # current_s tracks sum(current_nums) to efficiently check if all elements are zero.
            current_s = s_initial

            # Simulation loop. Iterates at most max_loop_iterations times.
            # Breaks earlier if curr goes out of bounds.
            for _ in range(max_loop_iterations):
                if not (0 <= curr < n):
                    # curr is out of bounds ([0, n-1] range), so the process ends.
                    break 
                
                if current_nums[curr] == 0:
                    curr += direction
                else: # current_nums[curr] > 0
                    current_nums[curr] -= 1
                    current_s -= 1      # Sum decreases by 1.
                    direction *= -1     # Reverse movement direction.
                    curr += direction   # Take a step in the new direction.
            
            # A selection is valid if, at the end of the process:
            # 1. All elements in nums have become 0 (i.e., current_s == 0).
            # 2. The process has ended, which means curr is out of bounds.
            #    (The loop terminates if curr is out of bounds, or if max_loop_iterations is reached.
            #     If max_loop_iterations is reached while curr is still in bounds,
            #     the second condition `not (0 <= curr < n)` will be false, correctly marking it invalid.)
            if current_s == 0 and not (0 <= curr < n):
                valid_selections_count += 1
                
    return valid_selections_count