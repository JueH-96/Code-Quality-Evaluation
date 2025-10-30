from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Constraints: 2 <= nums.length <= 100. 
        # This means n is guaranteed to be at least 2.
        # No explicit handling for n < 2 is needed due to these constraints.
        
        # Initialize max_abs_diff. Since absolute differences are non-negative,
        # 0 is a safe starting point. Any valid difference found will be >= 0.
        max_abs_diff = 0
        
        # Iterate through all elements. For each element nums[i],
        # its "next" adjacent element in a circular fashion is nums[(i + 1) % n].
        for i in range(n):
            # Calculate the absolute difference between nums[i] and nums[(i+1)%n].
            # The expression (i + 1) % n handles the wrap-around:
            # - For i from 0 to n-2, (i+1)%n is simply i+1.
            # - For i = n-1 (last element), (n-1+1)%n = n%n = 0. 
            #   This pairs the last element (nums[n-1]) with the first element (nums[0]).
            diff = abs(nums[i] - nums[(i + 1) % n])
            
            # If this difference is greater than the maximum found so far, update the maximum.
            if diff > max_abs_diff:
                max_abs_diff = diff
                
        return max_abs_diff