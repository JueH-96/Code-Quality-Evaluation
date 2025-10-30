import math

class Solution:
  def minimumArrayLength(self, nums: list[int]) -> int:
    # According to constraints, nums.length >= 1.
    # All elements are positive integers.

    # Step 1: Find the minimum value in the array.
    # Python's min() on a list is efficient (O(N) single pass).
    min_val = min(nums) 
            
    # Step 2: Check if this minimum value divides all other numbers in the array.
    # If min_val divides all numbers, then min_val is the GCD of the array.
    # If min_val does not divide at least one number, then the GCD of the array
    # (let's call it g) must be strictly smaller than min_val (g < min_val).
    # In this case (g < min_val), the minimum array length is 1.
    for x in nums:
      if x % min_val != 0:
        return 1
        
    # Step 3: If we reach this point, min_val divides all numbers in nums.
    # This means min_val is the GCD of all numbers in nums.
    # Now, we need to count how many times min_val appears in the array.
    count_min_val = 0
    for x in nums:
      if x == min_val:
        count_min_val += 1
        
    # When min_val is the GCD, the final length is ceil(count_min_val / 2).
    # `ceil(N/2)` can be written as `(N + 1) // 2` using integer division.
    return (count_min_val + 1) // 2