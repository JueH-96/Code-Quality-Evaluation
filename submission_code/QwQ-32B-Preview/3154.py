from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0  # Not enough elements to form a triplet
        
        # Precompute max_left for each j
        max_left = [0] * n
        max_left[1] = nums[0]
        for j in range(2, n):
            max_left[j] = max(max_left[j-1], nums[j-1])
        
        # Precompute max_right for each j
        max_right = [0] * n
        max_right[n-2] = nums[n-1]
        for j in range(n-3, -1, -1):
            max_right[j] = max(max_right[j+1], nums[j+1])
        
        # Compute the maximum triplet value
        max_value = float('-inf')
        for j in range(1, n-1):
            value = (max_left[j] - nums[j]) * max_right[j]
            if value > max_value:
                max_value = value
        
        return max_value if max_value > 0 else 0