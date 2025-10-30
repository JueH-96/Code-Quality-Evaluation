from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        total_valid = 0
        
        # For each starting index with nums[i] == 0
        for start in range(n):
            if nums[start] != 0:
                continue
            
            # Try both initial directions: -1 for left, +1 for right
            for initial_dir in (-1, 1):
                curr = start
                direction = initial_dir
                # Work on a copy so we don't mutate the original
                arr = nums[:]  
                
                # Simulate until we go out of bounds
                while 0 <= curr < n:
                    if arr[curr] == 0:
                        # just move in the current direction
                        curr += direction
                    else:
                        # decrement, reverse, then move
                        arr[curr] -= 1
                        direction = -direction
                        curr += direction
                
                # If all elements are zero, it's a valid selection
                if all(x == 0 for x in arr):
                    total_valid += 1
        
        return total_valid