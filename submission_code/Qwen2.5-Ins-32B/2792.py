from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # Check if the XOR of all elements in derived is 0
        return sum(derived) % 2 == 0