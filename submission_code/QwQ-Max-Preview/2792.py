from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        for start in [0, 1]:
            current = start
            for i in range(n - 1):
                next_val = current ^ derived[i]
                current = next_val
            if (current ^ start) == derived[-1]:
                return True
        return False