import sys
from typing import List

# Setting a higher recursion depth is typically for recursive solutions,
# this solution is iterative, so it's likely not necessary.
# sys.setrecursionlimit(2000) 

class Solution:
    """
    Solves the problem by simulating the process for each possible starting configuration.
    A starting configuration is defined by an initial position `curr` where nums[curr] == 0,
    and an initial movement direction (left or right).
    The simulation follows the rules described: move on 0, decrement/reverse/move on >0.
    A configuration is valid if the simulation ends with all elements of nums becoming 0.
    The count of valid configurations is returned.

    The approach involves:
    1. Identifying all possible starting indices (where nums[i] == 0).
    2. For each starting index, simulating the process twice: once starting left, once starting right.
    3. Each simulation runs on a copy of the original array to avoid interference.
    4. A step limit is used to detect potential infinite loops or overly long simulations.
    5. If a simulation ends with the array elements all being zero, the starting configuration is counted as valid.
    """
    def countValidSelections(self, nums: List[int]) -> int:
        """
        Counts the number of valid starting selections (position and direction).
        
        Args:
            nums: The input integer array.
            
        Returns:
            The number of valid selections.
        """
        
        n = len(nums)
        # Find all indices where nums[i] == 0. These are the only valid starting positions.
        zero_indices = [i for i, x in enumerate(nums) if x == 0]
        
        # Calculate the initial sum of elements and check if all are already zero.
        initial_sum = 0
        is_all_zero = True
        for x in nums:
             initial_sum += x
             if x != 0:
                is_all_zero = False
             
        # Optimization: If the array initially contains only zeros.
        if is_all_zero: # This check is equivalent to initial_sum == 0 since nums[i] >= 0.
            # In this case, any start from a 0 position will eventually move out of bounds
            # without modifying the array (since all elements are 0).
            # The array remains all zeros, so all possible starting selections are valid.
            # Each zero index allows two starting directions (left/right).
            return 2 * len(zero_indices)
            
        valid_count = 0
        
        # Set a maximum number of steps for the simulation to prevent infinite loops
        # or excessively long runs that could lead to Time Limit Exceeded errors.
        # The constraints N <= 100 and nums[i] <= 100 imply a maximum initial sum S <= 100*100 = 10000.
        # A rough upper bound on simulation steps is S * (N+1), which is about 10^6.
        # We use a fixed large constant, 2*10^6, as a safety margin.
        max_steps = 2 * 10**6 

        # Iterate through each potential starting index `start_idx` identified earlier.
        for start_idx in zero_indices:
            # For each starting index, there are two possible initial directions: left and right.

            # Simulate the process starting at `start_idx` with initial direction left (-1).
            # The `_simulate` method returns True if this configuration is valid.
            if self._simulate(nums, start_idx, -1, max_steps):
                valid_count += 1
            
            # Simulate the process starting at `start_idx` with initial direction right (1).
            # The `_simulate` method returns True if this configuration is valid.
            if self._simulate(nums, start_idx, 1, max_steps):
                valid_count += 1
                
        # After checking all possible starting configurations, return the total count of valid ones.
        return valid_count

    def _simulate(self, nums_orig: List[int], start_curr: int, start_dir_val: int, max_steps: int) -> bool:
        """
        Helper method to perform one simulation run for a given starting configuration.
        It simulates the process described in the problem statement and checks if the final state
        of the array consists of all zeros.
        
        Args:
            nums_orig: The original input array. A copy is made internally for simulation.
            start_curr: The starting index for the simulation.
            start_dir_val: The starting direction (-1 for left, 1 for right).
            max_steps: The maximum allowed steps for the simulation to prevent excessive runtime.
            
        Returns:
            True if the simulation terminates successfully with all array elements reduced to 0.
            False if the simulation exceeds max_steps or terminates with non-zero elements remaining.
        """
        n = len(nums_orig)
        # Create a mutable copy of the list for this simulation run.
        # This ensures that modifications within this simulation do not affect
        # the original list or subsequent simulations.
        nums = list(nums_orig) 
        curr = start_curr
        
        # Set the initial direction based on start_dir_val.
        # Direction: 1 indicates moving right, -1 indicates moving left.
        direction = start_dir_val 
        
        steps = 0 # Initialize step counter for this simulation run.

        # The main loop of the simulation continues as long as the current index `curr` 
        # is within the valid array bounds [0, n-1].
        while 0 <= curr < n:
            # Safety check: If the number of steps exceeds the predefined limit, stop the simulation.
            # This is crucial to prevent infinite loops or runs that are too long for typical time limits.
            if steps > max_steps:
                 # Exceeding max_steps implies the configuration is considered invalid.
                return False 

            # Apply the simulation rules based on the value at the current position `nums[curr]`.
            if nums[curr] == 0:
                # If the element is 0, simply move one step in the current direction.
                curr += direction
            else: # nums[curr] > 0
                # If the element is positive:
                nums[curr] -= 1   # Decrement the element's value by 1.
                direction *= -1   # Reverse the direction of movement (left becomes right, right becomes left).
                curr += direction # Take one step in the *new* direction.
            
            steps += 1 # Increment the step counter after each action (move or decrement/reverse/move).

        # The simulation loop terminates when `curr` moves out of the array bounds (curr < 0 or curr >= n).
        # After termination, we must check if the final state of the `nums` array consists entirely of zeros.
        
        # Iterate through the final state of the `nums` array used in this simulation.
        for x in nums:
            if x != 0:
                # If any element is found to be non-zero, this means the condition
                # "every element in nums becomes 0" is not met. Thus, this selection is not valid.
                return False
        
        # If the loop completes without finding any non-zero element, it means all elements are zero.
        # The condition is met, so this starting selection is valid.
        return True