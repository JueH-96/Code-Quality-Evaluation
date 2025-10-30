from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        # Precompute max_left[j] which is the maximum of nums[0..j-1]
        max_left = [0] * n
        current_max = nums[0]
        for j in range(1, n):
            max_left[j] = current_max
            if j < n - 1:
                current_max = max(current_max, nums[j])
        
        # Precompute max_right[j] which is the maximum of nums[j+1..n-1]
        max_right = [0] * n
        current_max = 0
        for j in range(n-2, -1, -1):
            current_max = max(nums[j+1], current_max)
            max_right[j] = current_max
        
        # Iterate over each valid j to compute the maximum value
        max_val = 0
        for j in range(1, n-1):
            current = (max_left[j] - nums[j]) * max_right[j]
            if current > max_val:
                max_val = current
        
        return max_val if max_val > 0 else 0