from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0

        def simulate_and_check(start_index: int, initial_direction: int, original_nums: List[int]) -> bool:
            """
            Simulates the process starting from start_index with initial_direction
            (-1 for left, 1 for right) on a copy of original_nums.
            Returns True if all numbers become zero, False otherwise.
            """
            current_nums = list(original_nums) # Work on a copy
            curr = start_index
            direction = initial_direction

            # Simulate until the current position is out of bounds
            # The process must terminate as sum(nums) decreases whenever hitting a positive number.
            # The total number of decrement steps is sum(original_nums).
            # Between decrement steps or before the first decrement, the pointer moves through zeros.
            # The number of steps between two decrement steps is bounded by N.
            # Total steps is bounded by O(Sum * N). Sum <= N * MaxVal.
            # Max steps ~ N^2 * MaxVal per simulation.
            # N=100, MaxVal=100 -> 100^2 * 100 = 10^6 steps max per simulation.
            # Total operations for all simulations ~ NumZeroIndices * 2 * N^2 * MaxVal.
            # NumZeroIndices <= N. Total ops ~ N * 2 * N^2 * MaxVal = O(N^3 * MaxVal).
            # For N=100, MaxVal=100, this is O(100^3 * 100) = O(10^8), which is acceptable.

            while 0 <= curr < n:
                # Read value at the current position
                val = current_nums[curr]

                if val == 0:
                    # If zero, move one step in the current direction
                    curr += direction
                    # Direction remains the same
                else: # val > 0
                    # If positive, decrement the value
                    current_nums[curr] -= 1
                    # Reverse the movement direction
                    direction = -direction
                    # Take a step in the new direction
                    curr += direction

            # After the process ends (curr is out of bounds), check if all elements are zero
            for val in current_nums:
                if val != 0:
                    # Found a non-zero element, simulation was not valid
                    return False

            # All elements are zero, simulation was valid
            return True

        # Find all valid starting positions (indices where nums[i] is 0)
        zero_indices = [i for i, val in enumerate(nums) if val == 0]

        # For each valid starting position, try both initial directions
        for start_index in zero_indices:
            # Try starting by moving left (-1)
            if simulate_and_check(start_index, -1, nums):
                count += 1
            # Try starting by moving right (1)
            if simulate_and_check(start_index, 1, nums):
                count += 1

        # Return the total number of valid selections
        return count