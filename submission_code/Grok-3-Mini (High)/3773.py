import math
from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        def is_non_decreasing(arr):
            for i in range(len(arr) - 1):
                if arr[i] > arr[i + 1]:
                    return False
            return True
        
        arr = list(nums)  # Make a copy of the input list
        count = 0
        
        while not is_non_decreasing(arr):
            # Find the index of the adjacent pair with the minimum sum
            min_sum = math.inf
            min_index = -1
            for i in range(len(arr) - 1):
                current_sum = arr[i] + arr[i + 1]
                if current_sum < min_sum:
                    min_sum = current_sum
                    min_index = i
            
            # Replace the pair with their sum
            arr[min_index] = min_sum  # Set the element at min_index to the sum
            del arr[min_index + 1]   # Remove the next element
            count += 1  # Increment the operation count
        
        return count