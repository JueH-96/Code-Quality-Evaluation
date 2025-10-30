from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        # Calculate the cumulative XOR of the derived array
        cumulative_xor = 0
        for i in range(n):
            cumulative_xor ^= derived[i]
        
        # If the cumulative XOR is 0, a valid original array exists
        return cumulative_xor == 0