from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # Calculate the total number of 1's in the derived array
        total_ones = sum(derived)
        
        # If the total number of 1's is odd, it's impossible to form a valid original array
        # Because XOR of two numbers is 1 only when one of them is 1 and the other is 0
        # So, the number of 1's in the original array must be even
        if total_ones % 2 == 1:
            return False
        
        # If the total number of 1's is even, it's possible to form a valid original array
        # We can always find a way to arrange the 1's in the original array to get the derived array
        return True