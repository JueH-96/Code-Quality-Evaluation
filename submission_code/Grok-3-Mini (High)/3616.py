from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        # Find all indices where nums[i] == 0
        zeros = [i for i in range(n) if nums[i] == 0]
        count = 0
        # For each starting position and each direction
        for start in zeros:
            for direction in [-1, 1]:  # -1 for left, 1 for right
                # Make a copy of nums to simulate without modifying original
                nums_copy = list(nums)
                curr = start
                step = direction  # Step direction, will be updated
                # Simulate the process until out of bounds
                while 0 <= curr < n:
                    if nums_copy[curr] == 0:
                        # If current cell is 0, just move in the current direction
                        curr += step
                    else:
                        # If current cell > 0, decrement it, reverse direction, and move
                        nums_copy[curr] -= 1
                        step = -step  # Reverse direction
                        curr += step  # Move in the new direction
                # After simulation, check if all elements are zero
                if all(num == 0 for num in nums_copy):
                    count += 1
        return count