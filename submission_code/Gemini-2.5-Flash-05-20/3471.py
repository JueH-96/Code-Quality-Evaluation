from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        # Step 1: Initialize an empty list to store the calculated averages.
        averages = []
        
        # Step 2: Sort the input array 'nums'.
        # This is crucial because it allows us to easily find the smallest
        # and largest elements by using pointers at the beginning and end.
        nums.sort()
        
        # Step 3: Initialize two pointers:
        # 'left' points to the smallest element.
        # 'right' points to the largest element.
        left = 0
        right = len(nums) - 1
        
        # Step 4: Iterate until the pointers cross each other.
        # Since 'n' is even, 'left' will eventually become greater than 'right'
        # after n/2 pairs have been processed.
        while left < right:
            # Get the current smallest and largest elements.
            min_element = nums[left]
            max_element = nums[right]
            
            # Calculate their average.
            current_average = (min_element + max_element) / 2
            
            # Add the calculated average to our list.
            averages.append(current_average)
            
            # Move the pointers inwards. This simulates removing the elements
            # from the 'nums' array for the next iteration.
            left += 1
            right -= 1
            
        # Step 5: After all pairs have been processed and their averages collected,
        # find and return the minimum value among these averages.
        return min(averages)