from typing import List
from itertools import permutations

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        # Use a set to avoid duplicate numbers
        unique_numbers = set()
        
        # Generate all permutations of the digits array of length 3
        for perm in permutations(digits, 3):
            # Convert tuple to a number
            num = perm[0] * 100 + perm[1] * 10 + perm[2]
            
            # Check if the number is a valid three-digit number (no leading zero) and even
            if num >= 100 and num % 2 == 0:
                unique_numbers.add(num)
        
        # Return the count of unique valid even numbers
        return len(unique_numbers)