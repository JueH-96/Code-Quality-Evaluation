from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of good numbers in an array.
        An element nums[i] is good if it's strictly greater than nums[i-k] and nums[i+k]
        (if those indices exist). If neither index exists, the element is good.
        
        Args:
            nums: A list of integers.
            k: An integer representing the distance to check.
            
        Returns:
            The sum of all good elements.
        """
        total_sum = 0
        n = len(nums)
        
        # Iterate through each element in the array
        for i in range(n):
            current_num = nums[i]
            
            # Condition check for i - k
            # Initialize as True, meaning condition is met if index doesn't exist
            prev_condition_met = True 
            prev_index = i - k
            
            # If the previous index i-k exists (within bounds [0, n-1])
            if prev_index >= 0:
                # The condition is met only if the current number is strictly greater
                if current_num <= nums[prev_index]:
                    prev_condition_met = False
            
            # Condition check for i + k
            # Initialize as True, meaning condition is met if index doesn't exist
            next_condition_met = True
            next_index = i + k
            
            # If the next index i+k exists (within bounds [0, n-1])
            if next_index < n:
                # The condition is met only if the current number is strictly greater
                if current_num <= nums[next_index]:
                    next_condition_met = False
            
            # An element nums[i] is considered good if conditions for both sides are met
            if prev_condition_met and next_condition_met:
                total_sum += current_num
                
        return total_sum