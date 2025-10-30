import bisect
from collections import defaultdict

class Solution:
    def numberOfSubarrays(self, nums: list) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute the next greater element indices
        next_greater = [n] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                next_greater[i] = stack[-1]
            else:
                next_greater[i] = n  # No next greater element
            stack.append(i)
        
        # Create a dictionary to store positions of each value
        pos_dict = defaultdict(list)
        for idx, val in enumerate(nums):
            pos_dict[val].append(idx)
        
        total = 0
        for i in range(n):
            M = nums[i]
            lst = pos_dict[M]
            
            # Find the left boundary (first index >= i)
            left = bisect.bisect_left(lst, i)
            
            # Find the right boundary (first index >= next_greater[i])
            right = bisect.bisect_left(lst, next_greater[i])
            
            # Add the number of valid j's
            total += (right - left)
        
        return total