from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        best = -1
        
        # We only form subarrays of length >= 2,
        # so we check each potential start index up to n-2
        for start in range(n - 1):
            # Check if we can start an alternating subarray here
            if nums[start + 1] - nums[start] == 1:
                length = 2
                # The next difference we expect is -1
                next_diff = -1
                
                # Try to extend the subarray
                for j in range(start + 1, n - 1):
                    if nums[j + 1] - nums[j] == next_diff:
                        length += 1
                        next_diff = -next_diff  # Flip between +1 and -1
                    else:
                        break
                
                best = max(best, length)
        
        return best