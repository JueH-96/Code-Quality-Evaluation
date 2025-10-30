from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')  # Initialize to infinity to find the minimum sum
        
        for j in range(1, n - 1):
            # Find the smallest nums[i] to the left of j where nums[i] < nums[j]
            left_min = float('inf')
            for i in range(j):
                if nums[i] < nums[j]:
                    if nums[i] < left_min:
                        left_min = nums[i]
            
            # Find the smallest nums[k] to the right of j where nums[k] < nums[j]
            right_min = float('inf')
            for k in range(j + 1, n):
                if nums[k] < nums[j]:
                    if nums[k] < right_min:
                        right_min = nums[k]
            
            # If both left_min and right_min are found, calculate the current sum
            if left_min != float('inf') and right_min != float('inf'):
                current_sum = left_min + nums[j] + right_min
                if current_sum < min_sum:
                    min_sum = current_sum
        
        # Return the minimum sum if found, otherwise return -1
        return min_sum if min_sum != float('inf') else -1