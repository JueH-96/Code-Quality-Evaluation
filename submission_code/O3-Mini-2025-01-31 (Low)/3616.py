from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        def simulate(start: int, direction: int) -> bool:
            # Create a working copy of nums
            arr = nums[:]
            n = len(arr)
            curr = start
            
            # Continue simulation until curr goes out of bounds
            while 0 <= curr < n:
                # If the current element is 0, simply move in the current direction.
                if arr[curr] == 0:
                    curr += direction
                else:
                    # Element > 0 case:
                    # decrement the element by 1
                    arr[curr] -= 1
                    # reverse the direction
                    direction = -direction
                    # move one step in the new direction
                    curr += direction
            # Simulation finished (curr is out-of-bound). Check if arr is all zero.
            return all(val == 0 for val in arr)
        
        valid_count = 0
        n = len(nums)
        
        # Only consider starting positions where nums[i] == 0.
        for i in range(n):
            if nums[i] == 0:
                # Try both movement directions: left (-1) and right (1).
                if simulate(i, -1):
                    valid_count += 1
                if simulate(i, 1):
                    valid_count += 1
        
        return valid_count