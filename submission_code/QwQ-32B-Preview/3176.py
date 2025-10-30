from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        left_min = [None] * n
        right_min = [None] * n
        min_sum = float('inf')
        
        # Compute left_min
        current_min = float('inf')
        for j in range(1, n):
            if nums[j-1] < nums[j]:
                if nums[j-1] < current_min:
                    current_min = nums[j-1]
                left_min[j] = current_min
            else:
                left_min[j] = current_min if current_min < nums[j] else None
        
        # Compute right_min
        current_min = float('inf')
        for j in range(n-2, -1, -1):
            if nums[j+1] < nums[j]:
                if nums[j+1] < current_min:
                    current_min = nums[j+1]
                right_min[j] = current_min
            else:
                right_min[j] = current_min if current_min < nums[j] else None
        
        # Find the minimum sum
        for j in range(1, n-1):
            if left_min[j] is not None and right_min[j] is not None:
                current_sum = left_min[j] + nums[j] + right_min[j]
                if current_sum < min_sum:
                    min_sum = current_sum
        
        return min_sum if min_sum != float('inf') else -1