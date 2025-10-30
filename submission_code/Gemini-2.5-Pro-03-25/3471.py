import math
from typing import List

class Solution:
    """
    Solves the problem of finding the minimum average formed by repeatedly 
    removing the smallest and largest elements from a list and averaging them.
    """
    def minimumAverage(self, nums: List[int]) -> float:
        """
        Calculates the minimum average according to the specified procedure:
        1. Repeatedly remove the smallest and largest elements from nums.
        2. Calculate their average.
        3. Store these averages.
        4. Return the minimum stored average.

        Args:
            nums: A list of integers with an even length (n).

        Returns:
            The minimum average calculated from the n/2 pairs.
        """
        
        # Step 1: Sort the input list `nums` in ascending order.
        # Sorting allows us to efficiently find the smallest (first element) 
        # and largest (last element) elements in each conceptual step, 
        # without actually removing them repeatedly which would be inefficient.
        nums.sort()
        
        # Step 2: Initialize a list to store the calculated averages.
        averages = []
        
        # Step 3: Get the length of the list. 
        # We know n is even and n >= 2 from the constraints.
        n = len(nums)
        
        # Step 4: Iterate n / 2 times. 
        # In each iteration `i`, we pair the i-th smallest element (at index `i` 
        # in the sorted list) with the i-th largest element (at index `n - 1 - i` 
        # in the sorted list). This simulates the process of removing the current 
        # minimum and maximum in each step.
        # The loop runs from i = 0 up to (n // 2) - 1.
        num_pairs = n // 2
        for i in range(num_pairs):
            # Get the i-th smallest element (conceptually the minimum in this step)
            minElement = nums[i]
            # Get the i-th largest element (conceptually the maximum in this step)
            # This is located at the corresponding position from the end of the sorted list.
            maxElement = nums[n - 1 - i]
            
            # Calculate the average. Use floating-point division (e.g., / 2.0 or 
            # rely on Python 3's default float division with /).
            avg = (minElement + maxElement) / 2.0
            
            # Add the calculated average to our list of averages.
            averages.append(avg)
            
        # Step 5: After calculating all n/2 averages, find the minimum value 
        # among them. The constraints guarantee that the `averages` list 
        # will contain at least one element (since n >= 2).
        min_average = min(averages)
        
        # Step 6: Return the overall minimum average found.
        return min_average