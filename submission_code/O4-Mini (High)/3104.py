from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        # Sort the array to make prefix checks easy
        nums.sort()
        n = len(nums)
        ans = 0
        
        # Check for k = 0: select no one
        # Condition: every student must be unhappy if selected (none are),
        # and every unselected student i must satisfy 0 < nums[i].
        if nums[0] > 0:
            ans += 1
        
        # Check for k from 1 to n-1
        # For a valid k, we need:
        #   nums[k-1] < k  (the k-th smallest is strictly less than k)
        # and
        #   nums[k] > k    (the (k+1)-th smallest is strictly greater than k)
        for k in range(1, n):
            if nums[k-1] < k and nums[k] > k:
                ans += 1
        
        # Check for k = n: select everyone
        # Condition: every selected student i must satisfy n > nums[i].
        # Since nums[i] < n by problem constraint, this is always true.
        # But we'll check explicitly against the last element.
        if nums[n-1] < n:
            ans += 1
        
        return ans