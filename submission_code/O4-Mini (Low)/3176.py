from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        best = float('inf')
        
        # For each possible peak index j
        for j in range(1, n-1):
            peak = nums[j]
            
            # Find the smallest nums[i] to the left of j such that nums[i] < peak
            left_min = float('inf')
            for i in range(j):
                if nums[i] < peak and nums[i] < left_min:
                    left_min = nums[i]
            
            # Find the smallest nums[k] to the right of j such that nums[k] < peak
            right_min = float('inf')
            for k in range(j+1, n):
                if nums[k] < peak and nums[k] < right_min:
                    right_min = nums[k]
            
            # If both a valid left and right exist, consider the sum
            if left_min < float('inf') and right_min < float('inf'):
                total = left_min + peak + right_min
                if total < best:
                    best = total
        
        # If we never updated best, there is no valid mountain triplet
        return best if best < float('inf') else -1