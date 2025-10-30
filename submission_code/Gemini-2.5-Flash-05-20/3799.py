import itertools
from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        
        # Use a set to store distinct valid numbers.
        # A set automatically handles duplicates, ensuring that each unique number
        # is counted only once.
        distinct_numbers = set()
        
        # Generate all possible permutations of 3 digits from the input list.
        # itertools.permutations correctly handles the case where the input 'digits'
        # list contains duplicate numbers, ensuring that each position in the permutation
        # is filled by a unique digit from the available pool at that step.
        for p in itertools.permutations(digits, 3):
            d1, d2, d3 = p
            
            # Constraint 1: The hundreds digit (d1) cannot be zero.
            # If it's zero, it's not a valid three-digit number.
            if d1 == 0:
                continue
            
            # Constraint 2: The units digit (d3) must be even.
            # An even number must end in 0, 2, 4, 6, or 8.
            if d3 % 2 != 0:
                continue
                
            # If both constraints are met, form the three-digit number.
            number = d1 * 100 + d2 * 10 + d3
            
            # Add the formed number to the set.
            distinct_numbers.add(number)
            
        # The total count of distinct valid numbers is the size of the set.
        return len(distinct_numbers)