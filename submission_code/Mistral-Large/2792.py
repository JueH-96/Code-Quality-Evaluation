from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        if n == 1:
            return True

        # Check the consistency of the derived array
        for i in range(n):
            if derived[i] != derived[0] ^ derived[(i + 1) % n]:
                return False

        return True