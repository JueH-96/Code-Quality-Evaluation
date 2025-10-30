from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        """
        We have derived[i] = original[i] XOR original[(i+1) % n].
        Taking the XOR of all derived elements:
            XOR(derived) = (a0^a1) ^ (a1^a2) ^ ... ^ (a_{n-1}^a0)
                         = a0 ^ a1 ^ a1 ^ a2 ^ ... ^ a_{n-1} ^ a0
                         = (a0^a0) ^ (a1^a1) ^ ... ^ (a_{n-1}^a_{n-1})
                         = 0
        A valid original exists if and only if XOR of all derived elements is 0.
        """
        total_xor = 0
        for bit in derived:
            total_xor ^= bit
        return total_xor == 0