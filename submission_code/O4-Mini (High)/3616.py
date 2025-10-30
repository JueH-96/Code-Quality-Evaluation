from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Try every starting zero position and both directions
        for start in range(n):
            if nums[start] != 0:
                continue
            
            for d in (-1, 1):
                arr = nums.copy()
                curr = start
                direction = d
                
                # Simulate until we move out of bounds
                while 0 <= curr < n:
                    if arr[curr] == 0:
                        curr += direction
                    else:
                        arr[curr] -= 1
                        direction = -direction
                        curr += direction
                
                # If all entries are zero at the end, it's valid
                if all(x == 0 for x in arr):
                    count += 1
        
        return count