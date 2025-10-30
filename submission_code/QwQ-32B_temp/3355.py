from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        if n < 2:
            return -1  # According to constraints, n >= 2, but handle just in case
        
        total = 0
        for p in possible:
            total += 1 if p else -1
        
        current_sum = 0
        for i in range(n):
            p = possible[i]
            current_sum += 1 if p else -1
            k = i + 1
            if k >= n:
                break
            if 2 * current_sum > total:
                return k
        
        return -1