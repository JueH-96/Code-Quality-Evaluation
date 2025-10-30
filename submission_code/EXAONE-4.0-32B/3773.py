from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        arr = nums[:]
        operations = 0
        
        def is_non_decreasing(arr):
            for i in range(1, len(arr)):
                if arr[i] < arr[i-1]:
                    return False
            return True
        
        while not is_non_decreasing(arr):
            min_sum = float('inf')
            min_index = -1
            n = len(arr)
            for i in range(n - 1):
                s = arr[i] + arr[i+1]
                if s < min_sum:
                    min_sum = s
                    min_index = i
            arr = arr[:min_index] + [min_sum] + arr[min_index+2:]
            operations += 1
            
        return operations