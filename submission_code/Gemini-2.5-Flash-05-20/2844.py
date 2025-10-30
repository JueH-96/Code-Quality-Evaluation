from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum_of_squares = 0
        
        # Iterate through all possible 1-indexed positions 'i' from 1 to n
        for i in range(1, n + 1):
            # Check if 'i' divides 'n' (the length of the array)
            if n % i == 0:
                # If 'i' divides 'n', the element at 1-indexed position 'i' is special.
                # In a 0-indexed Python list, this corresponds to index 'i - 1'.
                special_element = nums[i - 1]
                
                # Add the square of the special element to the total sum
                total_sum_of_squares += special_element * special_element
                
        return total_sum_of_squares