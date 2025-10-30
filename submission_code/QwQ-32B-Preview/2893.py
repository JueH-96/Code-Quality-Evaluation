from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        import math
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize based on the parity of nums[0]
        if nums[0] % 2 == 0:
            dp_even = nums[0]
            dp_odd = float('-inf')
        else:
            dp_even = float('-inf')
            dp_odd = nums[0]
        
        # Iterate through the array starting from the second element
        for i in range(1, n):
            if nums[i] % 2 == 0:
                new_dp_even = max(dp_even + nums[i], dp_odd + nums[i] - x)
                new_dp_odd = dp_odd
            else:
                new_dp_odd = max(dp_odd + nums[i], dp_even + nums[i] - x)
                new_dp_even = dp_even
            # Update the dp variables
            dp_even, dp_odd = new_dp_even, new_dp_odd
        
        # Return the maximum of dp_even and dp_odd
        return max(dp_even, dp_odd)