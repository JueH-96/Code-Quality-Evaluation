from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        result = 0
        
        # All indices that are allowed as a starting point
        zeros = [i for i, v in enumerate(nums) if v == 0]
        
        for start in zeros:
            for initial_dir in (-1, 1):          # -1 → left,  +1 → right
                arr = nums[:]                    # work on a copy
                curr = start
                direction = initial_dir
                
                # Run the described process
                while 0 <= curr < n:
                    if arr[curr] == 0:
                        curr += direction
                    else:                        # arr[curr] > 0
                        arr[curr] -= 1
                        direction *= -1          # reverse direction
                        curr += direction        # take one step in new direction
                
                # Check whether all numbers became zero
                if all(v == 0 for v in arr):
                    result += 1
                    
        return result