from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # The key insight is that the array can be reduced to a length of 1 or 2.
        # If there is a 1 in the array, we can always reduce the array to length 1.
        # Otherwise, we can reduce the array to length 2.
        
        if 1 in nums:
            return 1
        else:
            return 2