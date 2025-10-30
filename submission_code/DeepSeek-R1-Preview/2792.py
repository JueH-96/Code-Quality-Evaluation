from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        # Try starting with 0
        current = 0
        for i in range(n - 1):
            current ^= derived[i]
        if (current ^ 0) == derived[-1]:
            return True
        # Try starting with 1
        current = 1
        for i in range(n - 1):
            current ^= derived[i]
        if (current ^ 1) == derived[-1]:
            return True
        return False