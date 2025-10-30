from typing import List

class Solution:
  def largestPerimeter(self, nums: List[int]) -> int:
    """
    Finds the largest possible perimeter of a polygon that can be formed from side lengths in nums.
    """
    # Sort the array to easily identify the longest side of any subset.
    nums.sort()
    
    # We will work backwards from the largest possible set of sides.
    # `current_perimeter` tracks the sum of the sides of the candidate polygon.
    # Initially, we consider all numbers.
    current_perimeter = sum(nums)
    n = len(nums)
    
    # We iterate from a polygon with n sides down to a polygon with 3 sides.
    # The index `i` points to the longest side of the current candidate set (nums[0...i]).
    # A polygon must have at least 3 sides, so i must be at least 2.
    for i in range(n - 1, 1, -1):
      # The longest side in the current set nums[0...i] is nums[i].
      longest_side = nums[i]
      
      # The sum of the other sides is the total sum of the set minus the longest side.
      sum_of_other_sides = current_perimeter - longest_side
      
      # Check if the polygon condition holds: the sum of the shorter sides
      # must be strictly greater than the longest side.
      if sum_of_other_sides > longest_side:
        # If it holds, we've found a valid polygon. Since we are iterating
        # from the largest possible sum downwards, this is the largest
        # possible perimeter.
        return current_perimeter
      else:
        # If the condition fails, the current set of sides cannot form a polygon.
        # We must discard the current longest side and try with the next smaller set.
        # We update `current_perimeter` by removing the side we just discarded.
        current_perimeter -= longest_side
            
    # If the loop completes, no subset of 3 or more sides could form a valid polygon.
    return -1