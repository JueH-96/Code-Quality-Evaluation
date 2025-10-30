from typing import List
import copy

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        # Helper simulation function
        def simulate(original: List[int], start: int, direction: int) -> bool:
            arr = original.copy()  # make a copy so that we don't modify the original list
            n = len(arr)
            curr = start
            d = direction  # current direction: 1 for right and -1 for left
            
            while 0 <= curr < n:
                if arr[curr] == 0:
                    # Just move in the current direction.
                    curr += d
                else:  # arr[curr] > 0
                    # Decrement the value and reverse direction before taking a step.
                    arr[curr] -= 1
                    d = -d  # reverse direction
                    curr += d
            # Check if every entry is zero after the simulation ends.
            return all(val == 0 for val in arr)
        
        n = len(nums)
        valid_count = 0
        # We choose a starting position where nums[curr] == 0.
        for i in range(n):
            if nums[i] == 0:
                # Two choices for initial movement: left (-1) and right (1)
                for d in (1, -1):
                    if simulate(nums, i, d):
                        valid_count += 1
        return valid_count