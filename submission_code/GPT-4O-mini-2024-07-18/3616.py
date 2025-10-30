from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        valid_count = 0
        
        # Function to simulate the process
        def simulate(start_index: int, direction: int) -> bool:
            curr = start_index
            nums_copy = nums[:]
            while 0 <= curr < n:
                if nums_copy[curr] == 0:
                    curr += direction
                else:
                    nums_copy[curr] -= 1
                    direction = -direction  # Reverse direction
                    curr += direction
            
            return all(x == 0 for x in nums_copy)
        
        # Find all starting positions where nums[curr] == 0
        for i in range(n):
            if nums[i] == 0:
                # Try moving left (direction = -1) and right (direction = 1)
                if simulate(i, -1):
                    valid_count += 1
                if simulate(i, 1):
                    valid_count += 1
        
        return valid_count