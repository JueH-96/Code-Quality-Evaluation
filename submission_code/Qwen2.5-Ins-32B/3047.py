from typing import List
import math

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(x):
            s = int(math.sqrt(x))
            return s * s == x
        
        n = len(nums)
        max_sum = max(nums)  # Initialize with the maximum single element
        
        for i in range(n):
            for j in range(i + 1, n):
                if is_perfect_square(nums[i] * nums[j]):
                    current_sum = nums[i] + nums[j]
                    max_sum = max(max_sum, current_sum)
                    
                    # Check for larger subsets
                    for k in range(j + 1, n):
                        if is_perfect_square(nums[i] * nums[k]) and is_perfect_square(nums[j] * nums[k]):
                            current_sum += nums[k]
                            max_sum = max(max_sum, current_sum)
        
        return max_sum