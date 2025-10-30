from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        
        operations = 0
        
        while True:
            # Step 1: Check if the array is non-decreasing.
            # An array with 0 or 1 elements is considered non-decreasing.
            # The loop `range(len(nums) - 1)` correctly handles these cases.
            is_non_decreasing = True
            for i in range(len(nums) - 1):
                if nums[i] > nums[i+1]:
                    is_non_decreasing = False
                    break
            
            if is_non_decreasing:
                # If the array is non-decreasing, the process is complete.
                break
            
            # If not non-decreasing, we must perform an operation.
            operations += 1
            
            # Step 2: Find the adjacent pair with the minimum sum.
            # Since the array is not non-decreasing, its length must be at least 2.
            min_sum = float('inf')
            min_index = -1
            
            for i in range(len(nums) - 1):
                current_sum = nums[i] + nums[i+1]
                
                # If we find a new minimum, update the sum and index.
                # The strict '<' ensures that in case of a tie for the minimum sum,
                # the first one encountered (the leftmost one) is chosen.
                if current_sum < min_sum:
                    min_sum = current_sum
                    min_index = i
            
            # Step 3: Replace the pair with their sum.
            # The element at min_index is replaced by the sum, and the
            # next element is removed.
            nums[min_index] = min_sum
            del nums[min_index + 1]
            
        return operations