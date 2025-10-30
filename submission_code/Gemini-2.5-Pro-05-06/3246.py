from typing import List

class Solution:
  def hasTrailingZeros(self, nums: List[int]) -> bool:
    even_count = 0
    for num in nums:
      # A number is even if its least significant bit is 0 in binary.
      # This can be checked using the modulo operator (num % 2 == 0)
      # or bitwise AND ( (num & 1) == 0 ).
      if (num % 2 == 0):
        even_count += 1
      
      # If we have found at least two even numbers:
      # We can select these two even numbers. Let them be e1 and e2.
      # Both e1 and e2 have LSB = 0.
      # Their bitwise OR, e1 | e2, will have LSB = (LSB(e1) | LSB(e2)) = (0 | 0) = 0.
      # An LSB of 0 means the number is even, which implies its binary representation
      # has at least one trailing zero.
      # Since we need to select "two or more" elements, selecting two even numbers fulfills the criteria.
      if even_count >= 2:
        return True
        
    # If the loop completes, it means even_count is 0 or 1.
    # In this case, we cannot select two or more elements that are all even.
    # As established in the reasoning: for the bitwise OR of selected elements
    # to be even (have a trailing zero), all selected elements must be even.
    # If we don't have at least two even numbers in `nums`, this is impossible.
    return False