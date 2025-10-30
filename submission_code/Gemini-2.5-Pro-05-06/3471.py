from typing import List

class Solution:
  def minimumAverage(self, nums: List[int]) -> float:
    """
    Calculates averages by repeatedly taking the smallest and largest elements
    from nums, averaging them, and adding to a list of averages.
    Returns the minimum of these calculated averages.
    """
    
    # Sort the input list in non-decreasing order.
    # This allows us to easily access the smallest element at the beginning
    # and the largest element at the end of the list at each step.
    nums.sort() # Sorts in-place
    
    # This list will store the calculated averages.
    averages = []
    
    # The problem states that 'n' (initial length of nums) is even.
    # We repeat the process n / 2 times. This loop continues as long as
    # 'nums' is not empty. In each iteration, two elements are removed.
    while nums: # Condition is true as long as the list is not empty
      # Remove the smallest element (from the left end of the sorted list).
      # list.pop(0) removes the element at index 0 and returns it.
      min_element = nums.pop(0)
      
      # Remove the largest element (from the right end of the sorted list).
      # list.pop() removes the last element and returns it.
      max_element = nums.pop() 
      
      # Calculate the average of these two elements.
      # Using 2.0 ensures floating-point division. In Python 3, the / operator
      # performs float division by default (e.g., 5 / 2 = 2.5).
      current_avg = (min_element + max_element) / 2.0
      
      # Add the calculated average to our list of averages.
      averages.append(current_avg)
      
    # After the loop, 'nums' will be empty and 'averages' will contain
    # n / 2 floating-point averages.
    # The problem constraints (2 <= n) ensure that 'averages' is not empty.
    # Return the minimum value found in the 'averages' list.
    return min(averages)