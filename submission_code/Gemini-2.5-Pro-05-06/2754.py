from typing import List
# math module is not strictly needed if float('-inf') is used instead of -math.inf

class Solution:
  def maxStrength(self, nums: List[int]) -> int:
    n = len(nums)
    # Initialize max_strength to negative infinity.
    # Since nums is guaranteed to be non-empty (1 <= nums.length),
    # max_strength will be updated to at least the value of one element
    # or product of elements from nums.
    max_strength = float('-inf') # Using float('-inf') avoids needing math import

    # Iterate through all possible non-empty subsets.
    # A bitmask `i` from 1 to (2^n - 1) can represent these subsets.
    # (1 << n) is 2^n. The loop goes from 1 up to, but not including, 2^n.
    for i in range(1, 1 << n):
        current_product = 1
        # For the current subset (represented by mask `i`), calculate its strength.
        for j in range(n):
            # Check if the j-th element (nums[j]) is part of the current subset.
            # This is true if the j-th bit in the mask `i` is set.
            if (i >> j) & 1:
                current_product *= nums[j]
        
        # Update max_strength if the current subset's strength is greater.
        if current_product > max_strength:
            max_strength = current_product
            
    # The problem implies the result should be an integer.
    # Since all nums[i] are integers, their product will be an integer.
    # max_strength will hold an integer value (possibly stored in a float variable
    # if initialized as float, but Python handles large integers seamlessly).
    # Casting to int() is safe and ensures the return type matches.
    return int(max_strength)