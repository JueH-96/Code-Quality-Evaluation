import math
from typing import List

class Solution:
  """
  Calculates the maximum strength of a non-empty group of students,
  where strength is the product of their scores.
  """
  def maxStrength(self, nums: List[int]) -> int:
    """
    Finds the maximum product of a non-empty subset of nums using a greedy approach.

    Args:
      nums: A list of integers representing student scores.

    Returns:
      The maximum strength achievable.
    """

    # Filter numbers into positive, negative lists and check if zero exists.
    pos = [x for x in nums if x > 0]
    neg = [x for x in nums if x < 0]
    has_zero = any(x == 0 for x in nums)

    # Handle edge cases where the maximum product might be 0 or a single negative number.
    # This situation occurs if there are no positive numbers and at most one negative number.
    if not pos:
        # If there are no positive numbers:
        if len(neg) == 0: 
            # Only zeros are present in the input array. The non-empty group must be [0].
            return 0
        elif len(neg) == 1: 
            # There's exactly one negative number, and possibly zeros.
            if has_zero: 
                # If zero exists, the maximum strength is 0 (by choosing the subset [0]).
                # Example: [-5, 0] -> max strength 0.
                return 0
            else: 
                # Only one negative number in total, no positives, no zeros.
                # Example: [-5] -> max strength -5.
                return neg[0]
        # If len(neg) > 1 and no positives, we proceed to the main logic below.
        # Example: [-2, -3] -> max strength 6.
        # Example: [-2, -3, -4] -> max strength 12.

    # Main logic for calculating the maximum product:
    # This part handles cases where there are positive numbers, or multiple negative numbers.
    max_prod = 1
    
    # Include all positive numbers in the product, as they increase strength.
    for x in pos:
        max_prod *= x
    
    # Sort negative numbers in ascending order.
    # Example: [-9, -5, -1]
    neg.sort() 

    # If the count of negative numbers is odd, we must exclude one negative number
    # to make the product positive (or maximally large if it stays negative).
    # We exclude the negative number with the smallest magnitude (largest value, closest to zero),
    # which is the last element after sorting.
    if len(neg) % 2 == 1:
        # Remove the last element (largest value negative number).
        # If neg initially had only 1 element, it becomes empty after this.
        neg = neg[:-1] 
        
    # Multiply by the selected negative numbers.
    # If the count was even, all negatives are included.
    # If the count was odd, all negatives except the one closest to zero are included.
    for x in neg:
        max_prod *= x
            
    # Return the calculated maximum product.
    return max_prod