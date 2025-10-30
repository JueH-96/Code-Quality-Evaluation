from typing import List # For using List as a type hint
import itertools     # For generating permutations

class Solution:
  def totalNumbers(self, digits: List[int]) -> int:
    # A set to store the distinct 3-digit even numbers found.
    # Using a set automatically handles duplicates. For example, if digits = [2,2,0]
    # and we consider permutations of distinct indices, (digits[0], digits[1], digits[2])
    # might be (2a, 2b, 0) and (digits[1], digits[0], digits[2]) might be (2b, 2a, 0).
    # Both form the number 220. The set ensures 220 is counted only once.
    found_numbers = set()
    
    # `itertools.permutations(digits, 3)` generates all possible ordered
    # selections of 3 elements from the `digits` list.
    # Each element from `digits` is chosen based on its position in the input list.
    # This correctly models the condition "Each copy of a digit can only be used
    # once per number". For instance, if digits = [1, 2, 2], a permutation like (1,2,2)
    # uses the '1' once and both '2's once.
    for p in itertools.permutations(digits, 3):
        # p is a tuple of 3 digits, e.g., (digit_val_1, digit_val_2, digit_val_3)
        d1 = p[0]  # Hundreds digit
        d2 = p[1]  # Tens digit
        d3 = p[2]  # Units digit
        
        # Condition 1: A 3-digit number cannot have a leading zero.
        if d1 == 0:
            continue
            
        # Condition 2: The number must be even.
        # An even number has its last digit (units digit) as an even number (0, 2, 4, 6, 8).
        # So, d3 must be divisible by 2.
        if d3 % 2 != 0: # If d3 is odd
            continue
            
        # If both conditions are met, form the 3-digit number.
        # Example: if p = (1, 2, 4), number = 1*100 + 2*10 + 4 = 124.
        number = d1 * 100 + d2 * 10 + d3
        found_numbers.add(number) # Add to set to count distinct numbers
            
    # The result is the count of distinct numbers found.
    return len(found_numbers)