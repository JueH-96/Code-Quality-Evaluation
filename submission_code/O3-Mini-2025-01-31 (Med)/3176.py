from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        # Iterate over every possible middle element (j)
        for j in range(1, n - 1):
            left_candidate = float('inf')
            right_candidate = float('inf')
            # Look for valid left index i for the current j
            for i in range(j):
                if nums[i] < nums[j]:
                    left_candidate = min(left_candidate, nums[i])
            # Look for valid right index k for the current j
            for k in range(j + 1, n):
                if nums[k] < nums[j]:
                    right_candidate = min(right_candidate, nums[k])
            # If both valid candidates were found, update the minimum sum
            if left_candidate != float('inf') and right_candidate != float('inf'):
                current_sum = left_candidate + nums[j] + right_candidate
                min_sum = min(min_sum, current_sum)
                
        return min_sum if min_sum != float('inf') else -1